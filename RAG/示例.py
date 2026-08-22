"""
RAG · 向量数据库 最小可运行实现(零依赖纯 Python 版)
==================================================

目标:不借助 numpy / FAISS / Chroma,用最朴素的 Python 代码,一步步展示
「向量数据库到底怎么建、查询时里面发生了什么」。

包含 2 个版本:
  【版本 1】ToyVectorDB   —— 纯 Python 手搓,每一行都能看懂
  【版本 2】LangChain + Chroma 真实写法 —— 作为对照(装了库再跑)

JS 类比:整个向量数据库本质就是
    const db = [ { id, vec: [0.1, -0.3, ...], text, meta }, ... ]
    function search(qVec, k) {
      // 对 db 里每条算 cosine(qVec, vec),排序取前 k
      return db
        .map(x => ({ ...x, score: cosine(qVec, x.vec) }))
        .sort((a,b) => b.score - a.score)
        .slice(0, k)
    }
"""

import math
import random
from pprint import pprint

# ─────────────────────────────────────────────
# 0. 一个稳定的「伪 Embedding 模型」(手搓版)
# ─────────────────────────────────────────────
# 真实场景下这步是调 sentence-transformers / OpenAI API,
# 这里用「字符哈希 + 累加 + 归一化」模拟:
#   1. 每个字符用固定 seed 哈希出一个 64 维的小向量
#   2. 一句话的向量 = 所有字向量按位相加
#   3. 除以模长(L2 归一化)
#
# 效果:相同字越多 → 向量越像 → 余弦相似度越高
# (虽然质量远不如真 embedding 模型,但足够演示流程)

VEC_DIM = 64                       # 向量维度,真模型一般 384~1024
_CHAR_VEC_CACHE: dict[str, list[float]] = {}


def _char_vec(ch: str) -> list[float]:
    """给单个字符生成一个固定的随机向量(用它自己做 seed,保证稳定)"""
    if ch in _CHAR_VEC_CACHE:
        return _CHAR_VEC_CACHE[ch]
    rng = random.Random(hash(ch) & 0xFFFFFFFF)  # 用字符 hash 当 seed
    vec = [rng.uniform(-1.0, 1.0) for _ in range(VEC_DIM)]
    _CHAR_VEC_CACHE[ch] = vec
    return vec


def fake_embed(text: str) -> list[float]:
    """
    把文本转成向量(手搓 embedding)。
    真场景:vec = sentence_transformers.encode(text)
    """
    vec = [0.0] * VEC_DIM
    for ch in text:
        cv = _char_vec(ch)
        for i in range(VEC_DIM):
            vec[i] += cv[i]
    # L2 归一化:除以模长,这样余弦相似度就等价于点积
    norm = math.sqrt(sum(v * v for v in vec)) or 1e-9
    return [v / norm for v in vec]


# ─────────────────────────────────────────────
# 1. 余弦相似度(点积,因为上面已经归一化)
# ─────────────────────────────────────────────

def cosine_sim(a: list[float], b: list[float]) -> float:
    """
    cos(a, b) = Σ(a[i]*b[i]) / (|a|*|b|)
    因为 fake_embed 已经做过 L2 归一化,|a|=|b|=1,
    所以直接点积就行。
    JS 类比:
      function cosine(a, b) {
        let s = 0;
        for (let i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
      }
    """
    return sum(ai * bi for ai, bi in zip(a, b))


# ─────────────────────────────────────────────
# 2. ToyVectorDB:手搓的向量数据库
# ─────────────────────────────────────────────
# 内部就是 3 个平行的 list:
#   self.ids        : ["chunk_0", "chunk_1", ...]
#   self.vectors    : [ [0.12, -0.03, ...], ... ]   ← 核心:float 数组
#   self.metas      : [ {"text": "...", "page":1}, ... ]

class ToyVectorDB:
    def __init__(self, dim: int = VEC_DIM):
        self.dim = dim
        self.ids: list[str] = []
        self.vectors: list[list[float]] = []
        self.metas: list[dict] = []

    # ---- 建库:加一条 ----
    def add(self, id_: str, text: str, meta: dict | None = None) -> None:
        """
        等价于 Chroma 的 add / FAISS 的 add_with_ids。
        内部做两件事:① text → vec  ② 把 (id, vec, meta) 存下来
        """
        vec = fake_embed(text)                                    # ②-1 embedding
        assert len(vec) == self.dim, f"维度不匹配:期望 {self.dim},实际 {len(vec)}"
        self.ids.append(id_)                                      # ②-2 写入
        self.vectors.append(vec)
        self.metas.append({"text": text, **(meta or {})})

    # ---- 建库:批量加 ----
    def add_many(self, texts_with_meta: list[tuple[str, str, dict | None]]) -> None:
        """批量:[(id, text, meta), ...]"""
        for idx, text, meta in texts_with_meta:
            self.add(idx, text, meta)

    # ---- 查询:Flat 暴力 Top-K ----
    def search(self, query_text: str, k: int = 3) -> list[dict]:
        """
        【查询阶段内部发生的事】⭐
        1. query_text  → 同一个 fake_embed → query_vec
        2. 对 self.vectors 的每条:score = cosine(query_vec, vec)
        3. 按 score 从大到小排序,取前 k 条
        4. 返回 [{id, score, text, meta}, ...]

        真·向量数据库(HNSW/IVF):第 2 步用索引加速,不是真的每条都算。
        这里 Flat 暴力 = FAISS IndexFlatL2 / Chroma 默认小数据集行为。
        """
        query_vec = fake_embed(query_text)                           # ①

        scored = []
        for i in range(len(self.vectors)):                           # ②
            score = cosine_sim(query_vec, self.vectors[i])
            scored.append({
                "id": self.ids[i],
                "score": round(score, 4),
                "text": self.metas[i]["text"],
                "meta": {k: v for k, v in self.metas[i].items() if k != "text"},
            })

        scored.sort(key=lambda x: x["score"], reverse=True)          # ③
        return scored[:k]                                            # ④

    # ---- 方便调试:看看内部长啥样 ----
    def inspect(self) -> None:
        print(f"\n{'='*60}")
        print(f"🧮 ToyVectorDB 内部状态(dim={self.dim}, n={len(self.ids)})")
        print(f"{'='*60}")
        for i, (id_, meta) in enumerate(zip(self.ids, self.metas)):
            # 只打印向量前 5 位,用省略号代替后面(64 维太长)
            vec_head = ", ".join(f"{v:.3f}" for v in self.vectors[i][:5])
            text_short = meta["text"][:30] + ("..." if len(meta["text"]) > 30 else "")
            print(f"  [{i}] id={id_}")
            print(f"       text : {text_short}")
            print(f"       vec  : [{vec_head}, ...]  ({len(self.vectors[i])} 维)")
            print(f"       meta : { {k:v for k,v in meta.items() if k!='text'} }")
        print(f"{'='*60}\n")


# ─────────────────────────────────────────────
# 3. 准备一些模拟文档(模拟 Chunk 切分后的结果)
# ─────────────────────────────────────────────

# 格式:[(chunk_id, 原文, {metadata})]
DOC_CHUNKS = [
    ("ch_001",
     "员工请假流程:提前 3 天在 OA 系统提交申请,经部门主管审批后生效。年假全年 10 天。",
     {"source": "员工手册.pdf", "page": 8, "section": "请假制度"}),

    ("ch_002",
     "费用报销流程:所有因公消费需保留发票,当月月底前在报销系统上传电子发票,财务 10 个工作日内打款。",
     {"source": "员工手册.pdf", "page": 15, "section": "报销制度"}),

    ("ch_003",
     "出差规定:国内差旅住宿标准一线城市不超过 500 元/晚,高铁选二等座。超过标准需上级特批。",
     {"source": "员工手册.pdf", "page": 22, "section": "出差制度"}),

    ("ch_004",
     "入职体检:新员工需在报到后一周内到指定三甲医院完成体检,费用由公司承担,凭发票报销。",
     {"source": "入职指南.docx", "page": 3, "section": "体检"}),

    ("ch_005",
     "会议室预订:通过公司内部预订系统选择时间和会议室,时长超过 2 小时需抄送部门负责人。",
     {"source": "办公指南.pdf", "page": 5, "section": "会议室"}),

    ("ch_006",
     "工资发放:每月 15 日发放上月工资,逢节假日顺延至下一工作日。工资条在 HR 系统中查看。",
     {"source": "员工手册.pdf", "page": 12, "section": "薪酬"}),
]


# ─────────────────────────────────────────────
# 4. 主流程:建库 → 查询 → 看结果
# ─────────────────────────────────────────────

def main():
    # ================= 4.1 创建 + 写入(建库) =================
    print("📚 开始建库:把 6 个文档 chunk 写进 ToyVectorDB")
    db = ToyVectorDB()
    db.add_many(DOC_CHUNKS)
    db.inspect()  # 看看内部到底存了啥

    # ================= 4.2 语义查询演示 =================
    queries = [
        "请假怎么申请?",                  # 语义近:ch_001 请假流程
        "我出差酒店能住多少钱一晚?",        # 语义近:ch_003 出差规定
        "发票报销怎么弄?",                 # 关键词「报销」字面匹配 ch_002,
                                          # 但「发票」和 ch_004 「体检报销」也有点像
        "什么时候发工资?",                 # 语义近:ch_006
        "我想订个 3 小时的会议室",          # 语义近:ch_005
    ]

    for q in queries:
        print(f"\n💬 查询: {q}")
        print("-" * 60)
        results = db.search(q, k=3)
        for r in results:
            print(f"   ✅ score={r['score']:.4f}  id={r['id']}  "
                  f"meta={r['meta']}")
            print(f"       原文: {r['text']}")
        print()


# ─────────────────────────────────────────────
# 5. 真实写法(装了库以后对比用)
# ─────────────────────────────────────────────
# 把下面 try-except 里的代码,理解为 ToyVectorDB 的「等价真实版」。
# 需要先:  pip install sentence-transformers chromadb 或 faiss-cpu

def real_world_lanchain_chroma_demo():
    """
    真实项目里你会这么写(LangChain + Chroma)。
    和 ToyVectorDB 对比:
        ToyVectorDB.add      ←→  Chroma.from_texts
        ToyVectorDB.search   ←→  db.similarity_search
    内部流程完全一样,只是 embedding 换成了真模型,检索换成 HNSW 加速。
    """
    try:
        from langchain.vectorstores import Chroma
        from langchain.embeddings import HuggingFaceEmbeddings
    except ImportError:
        print("[真实写法跳过] 没装 langchain / chromadb / sentence-transformers,"
              " pip install 后就能运行")
        return

    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    texts = [x[1] for x in DOC_CHUNKS]
    metas = [x[2] for x in DOC_CHUNKS]

    # 建库 + 持久化到磁盘(下次 Chroma(..., embedding_function=emb) 直接 load)
    db = Chroma.from_texts(
        texts=texts,
        embedding=emb,
        metadatas=metas,
        persist_directory="./chroma_db",
    )

    docs = db.similarity_search("请假怎么申请?", k=3)
    for d in docs:
        print(d.page_content, d.metadata)


if __name__ == "__main__":
    main()
    print("\n" + "=" * 60)
    print("📦 下面是真实 LangChain + Chroma 的写法对照:")
    print("=" * 60)
    real_world_lanchain_chroma_demo()
