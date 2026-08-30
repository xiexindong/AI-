# 04 · search / query / delete —— 查询三件套怎么用

> 对应 [示例.py](../示例.py) 的 **Step 5**。回答的问题:「查数据有几种姿势,分别怎么操作的?」

---

## 一、一句话理解

| 方法 | 一句话 | SQL 类比 |
|---|---|---|
| **search** | 给一个向量,找最像的 Top-K | 无(语义检索特有) |
| **search + filter** | 在圈定范围内找最像的 Top-K | `WHERE + 语义排序` |
| **query** | 给标量条件,捞符合条件的行 | `SELECT ... WHERE` |
| **get** | 按主键精确拿一行 | `SELECT ... WHERE id=?` |
| **delete** | 按条件删(标记删除) | `DELETE ... WHERE` |

---

## 二、search:怎么操作的(4 步)

```python
q_vec = fake_embed("怎么申请年假?")          # ① 问题文本 → 向量(同一个模型!)
results = client.search(
    collection_name=COLLECTION,
    data=[q_vec],                            # ② 查询向量(可一次多条)
    limit=2,                                 # ③ Top-K
    output_fields=["text", "source", "page"],# ④ 命中后带回哪些列
)

for hit in results[0]:                       # results[0] = 第一条查询的结果列表
    print(hit["distance"], hit["entity"]["text"])
```

内部流程:

```
① 问题过 Embedding 模型(必须和建库同一个,否则向量空间不同全乱)
② Milvus 在 HNSW 图上从入口点贪心走:每层找最近邻居跳过去,
   不再改进就下一层,底层收 Top-K(见向量数据库/知识点.md 第五节)
③ 按 output_fields 回表取 text/source/page
④ 返回结构:results[第几条查询][第几个命中] → {distance, entity}
```

**坑**:`data=[q_vec]` 传的是列表,所以取结果永远先 `results[0]`;distance 在 COSINE 度量下越小越像(排序已排好)。

---

## 三、search + filter:Agent 生产的刚需

```python
results = client.search(
    collection_name=COLLECTION,
    data=[fake_embed("补办门禁要多少钱?")],
    limit=2,
    filter='source == "行政制度.docx"',   # 标量过滤:先圈地,再找近邻
    output_fields=["text", "source"],
)
```

- **执行顺序**:先按标量条件把候选数据圈定(比如只留某文档/某用户),再在圈内跑 HNSW——**不是查完再筛**,所以性能可控;
- **Agent 场景**:多用户记忆隔离 `filter='user_id == "u1001"'`,保证 A 用户永远检索不到 B 用户的记忆,这是 Agent 记忆系统的安全底线;
- **filter 语法**:字符串值用双引号 `source == "x"`,数字不用引号 `page >= 10`,组合用 `and / or`。

---

## 四、query:纯标量过滤

```python
rows_hit = client.query(
    collection_name=COLLECTION,
    filter="page >= 10",                        # 不给向量,纯条件
    output_fields=["text", "source", "page"],
)

# get:按主键精确拿,最快
client.get(collection_name=COLLECTION, ids=[456789])
```

什么时候用 query 而不是 search:**你关心「哪些数据满足条件」,不关心「和谁语义相近」**。比如运维排查「page>=10 的都删了没」「某 doc_id 的 chunk 都长什么样」。

---

## 五、delete:标记删除,不是立刻物理删

```python
client.delete(collection_name=COLLECTION, filter="page == 3")
```

内部发生了什么:

```
delete → 只写一条「删除标记」(delta log)
       → 查询时把带标记的数据过滤掉(看起来像删了)
       → 磁盘上的段、HNSW 图里的点,原封不动
       → 后台 compact(整理)时才真正物理清除
```

通俗类比:图书馆的书被借走,管理员先贴张「已借出」的条(查询时自动跳过),书和目录卡片都还在架上;等到整理书架(compact)那天才把卡片抽掉。

**坑**:大量删除后内存不会立刻释放(数据还在 QueryNode 里),要 release + load 或等 compact——这也是面试判断「有没有真用过」的细节题。

---

## 六、search vs query 对比(必考概念题)

| | search | query |
|---|---|---|
| 输入 | 查询向量 | 标量条件 |
| 走什么 | HNSW 图索引 | 标量索引/逐段过滤 |
| 回答的问题 | 「谁和这句话**最像**」 | 「哪些数据**满足条件**」 |
| 类比 | 以图搜图 | SQL select |
| 典型场景 | RAG 召回、Agent 记忆检索 | 数据管理、按 user_id 隔离排查 |

一句话总结:**search 管「像不像」,query 管「是不是」;生产 Agent 两者都离不开——query/filter 圈范围,search 找答案。**

---

## 七、面试要点(背这段)

> 查询三件套:search 传查询向量走 HNSW 取 Top-K,问题必须过和建库同一个 Embedding 模型;search+filter 先按标量圈定范围再找近邻,Agent 用它做 user_id 级记忆隔离;query 是纯标量条件过滤,相当于 SQL select;delete 是标记删除,写 delta log,查询时过滤,compact 才物理清除——所以大量删除后要 compact 或重新 load 才真正释放资源。

---

## 八、常见追问

**Q1:search 和 query 的区别?(100% 会问)**
见第六节对比表。补一句加分项:「我们 Agent 里两者组合:query/filter 圈定该用户的数据,search 在里面做语义召回。」

**Q2:limit=2 只返回 2 条,我想拿 20 条再 rerank 怎么办?**
limit 调大(如 20),再加 `ef` 参数(search 的 params={"ef": 128})保证 HNSW 候选队列 ≥ limit,否则召回不够。

**Q3:怎么保证 A 用户查不到 B 用户的记忆?**
两道防线:应用层在 filter 里强制拼 user_id;数据库层用 Partition 按 user_id 分区 + RBAC 权限。只靠应用层拼参数是防不住越权的,这是安全加分点。
