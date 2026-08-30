# 02 · insert 写入 —— 数据是怎么存进去的

> 对应 [示例.py](../示例.py) 的 **Step 3**。回答的问题:「insert 之后,Milvus 内部把我的数据怎么处理了?」

---

## 一、一句话理解

insert = 把「向量 + metadata」打包成一行 dict 交给 Milvus。你只管给数据,内部自动走一条流水线:

**记流水账(WAL)→ 攒成段(Segment)→ 落盘 → 异步建索引**。

Embedding 在 Milvus 之外算好——Milvus 只存向量、查向量,不负责向量化。

---

## 二、代码回顾

```python
rows = [
    {
        "vector": fake_embed(d["text"]),   # 文本 → 向量(在 Milvus 之外算好)
        "text": d["text"],                 # 原文一起存,召回后直接返回给 LLM
        "source": d["source"],             # metadata:出处,可追溯
        "page": d["page"],
    }
    for d in docs
]

res = client.insert(collection_name=COLLECTION, data=rows)
print(res["ids"])   # auto_id=True 时,这里返回 Milvus 自动生成的主键列表
```

---

## 三、写入内部流程(面试核心,讲这张图)

```
client.insert(rows)
   │
   ▼
Proxy 接入层:校验 Schema(维度对不对、字段缺不缺)
   │
   ▼
按主键哈希路由到某个 Shard(逻辑分片,决定并行度)
   │
   ▼
写 WAL(消息队列 Kafka/Pulsar)              ← 第 1 步:先记日志,保证不丢
   │
   ▼
DataNode 消费日志,写进 Growing Segment      ← 第 2 步:内存里的小段,可查未索引
   │
   ▼
攒够阈值(默认约 512MB 或超时)→ Seal 封段    ← 第 3 步:段变只读
   │
   ▼
Flush:整段写进对象存储 MinIO/S3             ← 第 4 步:持久化
   │
   ▼
IndexNode 异步建 HNSW 索引(不阻塞写入)      ← 第 5 步:做目录
```

---

## 四、Segment:看懂写入的钥匙

Milvus 不是一行一行地存,而是把数据攒成一段一段(Segment):

| 段状态 | 在哪 | 有没有索引 | 能不能查 |
|---|---|---|---|
| **Growing**(生长中) | 内存 | 没有 | 能(暴力算,慢一点) |
| **Sealed**(已封存) | 内存 + 待落盘 | 建索引中 | 能 |
| **Flushed**(已落盘) | 对象存储 | 有 | 能(load 后) |

通俗类比:外卖订单先写在门口小黑板(Growing,随时能改能查),写满一页撕下来装订(Sealed),装订册放进档案柜(Flushed),最后给每册做目录(索引)——**任何时候来查订单都查得到,只是慢快的区别**。

这也回答了经典追问「insert 完立刻查,查得到吗?」——**查得到**,走 Growing 段暴力算;默认一致性 Bounded,容忍毫秒级延迟换性能。

---

## 五、为什么生产都批量 insert

- 一批 → 一个段;一条一条插 → 产生海量碎片小段 → 段数量爆炸 → 索引多、查询要扫的段多、后台疯狂合并(compaction);
- **经验**:几百到几千条一批,配合并发,吞吐能差出几十倍;
- 示例里 4 条一次 insert 就是这个道理,生产是「全量文档分批灌库」的离线脚本。

---

## 六、面试要点(背这段)

> insert 内部五步:Proxy 校验 → 主键哈希路由到 Shard → 先写 WAL 保证不丢 → DataNode 写进内存 Growing Segment(无索引也能查)→ 攒满封段 Seal 落对象存储,IndexNode 异步建 HNSW。所以写入和建索引是异步解耦的,insert 返回成功只代表日志记完,Growing 段即可查(Bounded 一致性)。

---

## 七、常见追问

**Q1:为什么先写 WAL 再写内存?**
宕机恢复靠重放日志。消息队列既当持久化日志,又当写入缓冲削峰,还把「写」和「处理」解耦(DataNode 挂了不丢数据)。

**Q2:批量大小怎么定?**
和 Segment 封段阈值(约 512MB)、内存、网络配合着调;常见从 1000 条/批起步压测,看吞吐和延迟曲线,不是拍脑袋。

**Q3:向量化为什么不让 Milvus 做?**
职责分离:Embedding 是模型计算(GPU/模型推理),Milvus 是存储检索,解耦后各自独立扩展。2.4+ 的 Function 字段只是把「调 embedding」这一步托管,本质没变。
