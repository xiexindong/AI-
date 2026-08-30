# 01 · 建 Collection —— 表和字段怎么定义

> 对应 [示例.py](../示例.py) 的 **Step 2**。回答的问题:「建 Collection 这一步,代码在干嘛,Milvus 内部发生了什么?」

---

## 一、一句话理解

建 Collection = 在 Milvus 里**建一张「向量表」**。MySQL 建表是一条 `CREATE TABLE`,Milvus 分两步:

1. 用 **Schema** 定义「有哪些列」(向量列 + 标量列);
2. 用 **index_params** 定义「向量列用什么索引」(HNSW)。

通俗类比:先画好仓库的货架图纸(Schema),同时决定货架用什么检索方式(索引),然后空仓库开张——这时候还没有货(数据)。

---

## 二、代码回顾

```python
# 2.1 已存在就删掉(示例可重复跑)
if client.has_collection(COLLECTION):
    client.drop_collection(COLLECTION)

# 2.2 定义 Schema:一张表的「列清单」
schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
schema.add_field("id",     DataType.INT64,        is_primary=True)   # 主键,自动生成
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)           # 向量列(核心)
schema.add_field("text",   DataType.VARCHAR, max_length=1024)        # chunk 原文
schema.add_field("source", DataType.VARCHAR, max_length=256)         # metadata:来源
schema.add_field("page",   DataType.INT32)                           # metadata:页码

# 2.3 向量列的索引参数
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 32, "efConstruction": 200},
)

# 2.4 建表 + 建索引
client.create_collection(collection_name=COLLECTION,
                         schema=schema, index_params=index_params)
```

---

## 三、5 个字段逐个拆(为什么每列都要)

| 字段 | 类型 | 作用 | MySQL 类比 |
|---|---|---|---|
| `id` | INT64 + 主键 | 每行唯一标识,删除/更新全靠它 | 自增主键 |
| `vector` | FLOAT_VECTOR, dim=48 | 存 embedding,维度**建完不可改** | 无对应(特有列) |
| `text` | VARCHAR | chunk 原文,召回后直接给 LLM 用 | text 列 |
| `source` | VARCHAR | metadata,做 filter 过滤 + 回答溯源 | varchar 列 |
| `page` | INT32 | metadata,定位到页 | int 列 |

**红线**:`dim` 必须和 Embedding 模型输出维度一致(bge-small-zh=512)。建完 Collection 就不能改维度,换模型 = 删表重建。

---

## 四、Schema 的两个开关

| 开关 | 含义 | 怎么选 |
|---|---|---|
| `auto_id=True` | 主键由 Milvus 自动生成,插入时不传 id | 知识库场景推荐;要自己控制 id(如 doc_id+chunk_id 拼接)就设 False |
| `enable_dynamic_field=True` | 允许插入 Schema 之外的字段 | 类比 MongoDB 的灵活字段;灵活但破坏约束,生产建议少用 |

---

## 五、快跑模式 vs 企业写法

```python
# 快跑模式(5 分钟 demo):
client.create_collection("demo", dimension=512)
# 自动:默认主键 auto_id、默认向量列名、自动 HNSW 索引
# 缺点:没有 metadata 列,做不了 filter,生产直接淘汰

# 企业写法(示例.py 用的):
# 显式 Schema + 显式 index_params → metadata 过滤、索引可调、字段可控
```

面试一句话:「快跑模式验证连通性,生产必须显式 Schema,因为 Agent 场景一定要按 user_id/doc_id 做 metadata 过滤。」

---

## 六、这一步 Milvus 内部发生了什么(怎么操作的)

```
client.create_collection()
   → Proxy(接入层)校验参数
   → RootCoord(协调器)把表结构写进 etcd(元数据存储)   ← 只登记,不碰数据
   → 预分配逻辑分片(Shard/VChannel)和分区结构
   → 返回成功
```

关键认知:**建完 Collection 时,一条数据都没有**,只是把「图纸」登记在案。数据是 insert 阶段才进来的(见 02)。

通俗类比:开店前先去工商局注册(元数据入 etcd),营业执照拿到了,但店里还没进货。

---

## 七、面试要点(背这段)

> 建 Collection 用显式 Schema:主键 id(auto_id 自动生成)、向量列(dim 和 Embedding 模型一致,建后不可改)、text 存原文、source/page 做 metadata。向量列配 HNSW 索引(M=32、efConstruction=200、COSINE 度量)。内部只做一件事——RootCoord 把元数据写进 etcd,数据是 insert 阶段才写入的。

---

## 八、常见追问

**Q1:向量维度为什么建完不能改?**
存储布局和索引(HNSW 图节点)都按 D 维排布,改维度等于全部重写,所以 Milvus 直接禁止,换模型就删表重建。

**Q2:VARCHAR 为什么必须写 max_length?**
Milvus 是定长预分配存储,不像 MySQL 可变长,所以建表时就要给定长上限,超长插入直接报错。

**Q3:Partition 和 Shard 什么区别?**
Partition 是**逻辑**切分(你按规则手动分,如按租户),查询时指定 partition_names 减少扫描;Shard 是**物理**并行通道(insert 按主键哈希自动路由,决定写入并发度),用户不感知。
