"""
Milvus · 完整可运行上手示例(pymilvus MilvusClient 新版 API)
============================================================

对照 知识点.md 的学习路径,跑通生产 5 件事:
  Step 1   连接 Milvus(Docker Standalone 或 Zilliz Cloud)
  Step 2   建 Collection(带 Schema:id + 向量 + 标量字段)
  Step 3   insert 写入向量 + metadata
  Step 4   建 HNSW 索引 + load
  Step 5   search(向量近邻 + metadata 过滤) / query / delete

环境准备(Windows 必看,二选一):
  方式 A:Docker Desktop 启动 standalone
     $ docker compose up -d     # 用官方 milvus-standalone-docker-compose.yml
     默认端口 http://localhost:19530
  方式 B:Zilliz Cloud 免费集群(Serverless),拿 uri + token
     代码零改动,只换 MILVUS_URI / MILVUS_TOKEN 两个常量

依赖安装:
  $ pip install pymilvus
"""

from pprint import pprint

from pymilvus import DataType, MilvusClient

# ─────────────────────────────────────────────
# 0. 连接参数(按你的环境改这里)
# ─────────────────────────────────────────────
# 本地 Docker standalone 的默认地址;Zilliz Cloud 就换成它的 https uri
MILVUS_URI = "http://localhost:19530"
# Zilliz Cloud 才需要 token;本地 standalone 留空即可
MILVUS_TOKEN = ""

# 向量维度:必须和「Embedding 模型输出维度」一致。
# 生产用 bge-small-zh-v1.5 → 512;这里为了和之前手搓的 fake_embed 保持一致用 48,
# 只影响存储与索引,不影响学习 Milvus 本身。
DIM = 48

# Collection 名字:相当于 MySQL 的表名
COLLECTION = "kb_demo"


# ─────────────────────────────────────────────
# 1. 手搓 Embedding(占位,生产换 bge 模型)
# ─────────────────────────────────────────────
# 【这段在干嘛】
# 把文本变成 DIM 维浮点列表。Milvus 只负责「存向量、查向量」,
# 「文本 → 向量」这一步永远发生在 Milvus 之外(调用 Embedding 模型)。
# 这里用固定 seed 的伪随机数模拟,让示例不依赖任何模型库就能跑。

import random

_EMB_SEED = 42
_CHAR_CACHE: dict[str, list[float]] = {}


def fake_embed(text: str) -> list[float]:
    """文本 → DIM 维向量(演示用,生产换 BAAI/bge-small-zh-v1.5)。"""
    rng = random.Random(_EMB_SEED)
    vec = [0.0] * DIM
    for ch in text:
        if ch not in _CHAR_CACHE:
            # 每个字一条固定「指纹」,重复出现直接复用
            _CHAR_CACHE[ch] = [rng.uniform(-1, 1) for _ in range(DIM)]
        for i in range(DIM):
            vec[i] += _CHAR_CACHE[ch][i]
    # L2 归一化:向量长度拉成 1,余弦相似度才准
    norm = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / norm for x in vec]


# ─────────────────────────────────────────────
# Step 1  连接 Milvus
# ─────────────────────────────────────────────
# 【这段在干嘛】
# 创建客户端 = 拿到数据库连接。之后所有操作都通过 client.xxx() 完成。
# JS 类比:const client = new MongoClient(uri)

client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
print("✅ 已连接:", MILVUS_URI)

# 连不上会在这里抛异常(检查 Docker 是否 up:docker ps | grep milvus)


# ─────────────────────────────────────────────
# Step 2  建 Collection(显式 Schema,企业写法)
# ─────────────────────────────────────────────
# 【这段在干嘛】
# Milvus 建「向量表」分两步:先定义 Schema(列),再定索引参数(方式)。
# 企业里都用显式 Schema,因为要带 metadata(来源、页码、user_id)做过滤。
# 对比:client.create_collection(COLLECTION, dimension=DIM) 是「快跑模式」,
#       自动建默认 schema + 自增 id,适合 5 分钟 demo,不适合生产。

# 2.1 清理旧表(重复运行示例不报错;生产慎用 drop)
if client.has_collection(COLLECTION):
    client.drop_collection(COLLECTION)

# 2.2 定义 Schema:一张表的「列清单」
schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
#   auto_id=True          → 主键让 Milvus 自动生成(我们不用手动传 id)
#   enable_dynamic_field  → 允许插入 schema 之外的字段(灵活但建议少用)

# 每行数据 = 1 个向量 + 它的原始文本 + metadata
schema.add_field("id", DataType.INT64, is_primary=True)          # 主键(自动生成)
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)       # 向量列(核心)
schema.add_field("text", DataType.VARCHAR, max_length=1024)      # chunk 原文
schema.add_field("source", DataType.VARCHAR, max_length=256)     # metadata:来源文档
schema.add_field("page", DataType.INT32)                         # metadata:页码

# 2.3 定义索引参数:向量列用 HNSW(默认首选,见知识点第五节)
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="HNSW",            # 分层小世界图,O(log N)
    metric_type="COSINE",         # 余弦相似度(文本检索标配)
    params={"M": 32, "efConstruction": 200},   # 邻居数/建图候选队列
)

# 2.4 建表 + 建索引
client.create_collection(
    collection_name=COLLECTION,
    schema=schema,
    index_params=index_params,
)
print("✅ Collection 已创建:", COLLECTION)


# ─────────────────────────────────────────────
# Step 3  insert 写入(向量 + metadata 一起存)
# ─────────────────────────────────────────────
# 【这段在干嘛】
# 每条数据 = 一个 dict,键名对应 Schema 字段。向量已经在外面算好,
# Milvus 只负责存储。生产中这里通常是「离线批处理脚本」灌全量文档。

docs = [
    {"text": "员工请假需要提前 3 天在 OA 系统提交申请", "source": "员工手册.pdf", "page": 8},
    {"text": "报销请在月底前上传发票,逾期顺延至下月",   "source": "员工手册.pdf", "page": 15},
    {"text": "出差酒店标准不超过每晚 500 元",           "source": "员工手册.pdf", "page": 22},
    {"text": "门禁卡遗失请到行政前台补办,费用 20 元",   "source": "行政制度.docx", "page": 3},
]

rows = [
    {
        "vector": fake_embed(d["text"]),   # 文本 → 向量(Embedding 在 Milvus 之外)
        "text": d["text"],                 # 原文一起存,召回后直接返回给 LLM
        "source": d["source"],             # metadata:告诉 LLM 出处,可追溯
        "page": d["page"],
    }
    for d in docs
]

res = client.insert(collection_name=COLLECTION, data=rows)
print("✅ 插入完成,自增主键:", res["ids"])
# 注意:auto_id=True 时返回里才有 ids;主键 id 我们没传,Milvus 自动分配


# ─────────────────────────────────────────────
# Step 4  load(Milvus 特色:先加载进内存才能查)
# ─────────────────────────────────────────────
# 【这段在干嘛】
# 数据持久化在对象存储(磁盘),search/query 走 QueryNode 内存,
# 所以查询前必须 load。这一步 MilvusClient 建表时通常已自动做,
# 显式写一遍是为了记住这个知识点(没 load 就 search 会报错)。

client.load_collection(COLLECTION)
print("✅ 已 load 进内存,可以查询")


# ─────────────────────────────────────────────
# Step 5  查询三件套:search / search+filter / query
# ─────────────────────────────────────────────

# 5.1 纯向量 search:给一个问题,找最像的 Top-K
# 【核心方法】问题和文档都要过「同一个 Embedding 模型」再比较
q_vec = fake_embed("怎么申请年假?")
results = client.search(
    collection_name=COLLECTION,
    data=[q_vec],                 # 查询向量列表(可一次多查)
    limit=2,                      # Top-K
    output_fields=["text", "source", "page"],   # 命中后带回哪些列
)
print("\n── 5.1 纯向量 search(问题:怎么申请年假?) ──")
for hit in results[0]:            # results[0] = 第一条查询向量的结果
    # distance = 距离,越小越像(COSINE 度量下也常显示为相似度排序)
    print(f"  {hit['distance']:.3f}  [{hit['entity']['source']} p{hit['entity']['page']}]",
          hit["entity"]["text"])

# 5.2 向量 search + 标量过滤(Agent 生产刚需)
# 【场景】只想在「员工手册.pdf」里找,别的文档不许混进来。
# filter 在 Top-K 计算前先圈定数据范围 = 向量检索 + 条件过滤混合查询。
results = client.search(
    collection_name=COLLECTION,
    data=[fake_embed("补办门禁要多少钱?")],
    limit=2,
    filter='source == "行政制度.docx"',          # 标量过滤:字符串用双引号
    output_fields=["text", "source"],
)
print("\n── 5.2 search + filter(只在行政制度.docx 里找) ──")
for hit in results[0]:
    print(f"  {hit['distance']:.3f}  [{hit['entity']['source']}]",
          hit["entity"]["text"])

# 5.3 query:纯标量条件过滤(不给向量,相当于 SQL 的 select)
rows_hit = client.query(
    collection_name=COLLECTION,
    filter="page >= 10",                          # 数字条件不用引号
    output_fields=["text", "source", "page"],
)
print("\n── 5.3 query(标量条件 page >= 10) ──")
for r in rows_hit:
    print(f"  [p{r['page']} {r['source']}] {r['text']}")

# 5.4 delete:按标量条件删(删除后 HNSW 图不会立即收缩,量大后要 compact)
client.delete(collection_name=COLLECTION, filter='page == 3')
print("\n✅ 已删除 page==3 的数据")


# ─────────────────────────────────────────────
# 收尾:统计 + 清理(想保留数据就注释掉 drop)
# ─────────────────────────────────────────────
stats = client.get_collection_stats(COLLECTION)
print("\n当前行数:", stats["row_count"])

# client.drop_collection(COLLECTION)   # 清理现场;想留着复习就注释掉
print("示例结束。下一步:把 Chroma 的 RAG demo 迁移到这份 Milvus 代码上。")
