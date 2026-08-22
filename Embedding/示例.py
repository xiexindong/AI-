"""
Embedding 最小可运行演示(零依赖纯 Python)
==========================================

3 个演示,对应知识点里的 3 条核心性质:
  ① 性质 1:同模型空间下,语义近 → 向量近(余弦相似度高)
  ② 性质 2:不同模型(不同 seed)→ 不同空间 → 相似度失效
  ③ 性质 3:Embedding 和向量数据库的协作流(配合 RAG 示例的 ToyVectorDB)
"""

import math
import random
from pprint import pprint


# ─────────────────────────────────────────────
# 1. 可配置 seed 的手搓 Embedding(关键:seed 变 = 模型变)
# ─────────────────────────────────────────────
# 原理:每个字符用 (seed+字符) 做随机种子生成 D 维小向量,整句话向量 = 求和后 L2 归一化
# 同样的 seed → 同样的字符向量 → 同一个「模型空间」
# 不同的 seed → 不同的字符向量 → 不同「模型空间」,跨空间比相似度无意义

DIM = 48
_CHAR_CACHE: dict[tuple, list[float]] = {}


def fake_embed(text: str, seed: int) -> list[float]:
    """
    手搓版 Embedding。
    参数 seed 代表「这是哪个模型」,seed 相同 = 同一个模型。
    真场景:vec = SentenceTransformer('bge-small').encode(text)
    """
    vec = [0.0] * DIM
    for ch in text:
        key = (seed, ch)
        if key not in _CHAR_CACHE:
            rng = random.Random(hash((seed, ord(ch))) & 0xFFFFFFFF)
            _CHAR_CACHE[key] = [rng.uniform(-1.0, 1.0) for _ in range(DIM)]
        cv = _CHAR_CACHE[key]
        for i in range(DIM):
            vec[i] += cv[i]
    # L2 归一化
    norm = math.sqrt(sum(v * v for v in vec)) or 1e-9
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """因为都归一化过,直接点积就是余弦相似度"""
    return round(sum(ai * bi for ai, bi in zip(a, b)), 4)


# ─────────────────────────────────────────────
# 2. 演示 ① 同模型空间,语义近 → 相似度高
# ─────────────────────────────────────────────

def demo_1_same_model_similarity():
    print("\n" + "=" * 70)
    print("📌 演示 ①:【同一个模型】下,「语义相近 ↔ 余弦相似度高」")
    print("=" * 70)

    SEED_A = 42  # 同一个模型
    texts = [
        ("猫咪",      "语义近 A"),
        ("小猫",      "语义近 B(应该和猫咪得分高)"),
        ("猫猫狗狗",  "语义中(也和动物相关)"),
        ("汽车",      "语义远(应该得分低)"),
        ("开小轿车",  "语义远 B(和汽车近,和猫咪远)"),
    ]

    vecs = [(label, desc, fake_embed(label, SEED_A)) for (label, desc) in texts]

    print(f"模型 seed = {SEED_A}, 向量维度 = {DIM}")
    print("\n【把每对文本的相似度打印成一个矩阵】:")
    # 表头
    header = f"{'':>10}" + "".join(f"{t[0]:>10}" for t in texts)
    print(header)
    for i, (lbl_i, _, vi) in enumerate(vecs):
        row = f"{lbl_i:>10}"
        for _, _, vj in vecs:
            s = cosine(vi, vj)
            # 只看下三角(不含对角线),对角线都是 1.0 没意义
            row += f"{s:>10.3f}"
        print(row)

    # 重点挑几对说
    pairs = [
        ("猫咪", "小猫",      "✅ 都讲猫,语义近"),
        ("猫咪", "猫猫狗狗",   "↔️ 都讲动物,有点近"),
        ("猫咪", "汽车",      "❌ 语义完全无关"),
        ("汽车", "开小轿车",   "✅ 都讲车,语义近"),
    ]
    vec_map = {lbl: vec for lbl, _, vec in vecs}
    print("\n【挑重点对】:")
    for a, b, note in pairs:
        print(f"  cos({a:>6}, {b:>8}) = {cosine(vec_map[a], vec_map[b]):.4f}   {note}")


# ─────────────────────────────────────────────
# 3. 演示 ② 不同模型(不同 seed)→ 相似度完全失效
# ─────────────────────────────────────────────

def demo_2_different_model_space():
    print("\n" + "=" * 70)
    print("📌 演示 ②:【不同模型空间】下的相似度完全没有意义(红线!)")
    print("=" * 70)

    text_a = "猫咪"
    text_b = "小猫"
    text_c = "汽车"

    # 同一个模型(seed=42),正常结果
    v42_a = fake_embed(text_a, 42)
    v42_b = fake_embed(text_b, 42)
    v42_c = fake_embed(text_c, 42)

    # 换模型(seed=10086)→ 完全不同的坐标系
    v100_a = fake_embed(text_a, 10086)
    v100_b = fake_embed(text_b, 10086)
    v100_c = fake_embed(text_c, 10086)

    def show(label, a, b):
        print(f"  cos({label:>22}) = {cosine(a, b):.4f}")

    print("\n✅ 同一个模型内:相似度正常(猫咪和小猫高,和汽车低)")
    show("seed=42  猫咪 ↔ 小猫",     v42_a, v42_b)
    show("seed=42  猫咪 ↔ 汽车",     v42_a, v42_c)
    show("seed=10086 猫咪 ↔ 小猫",   v100_a, v100_b)
    show("seed=10086 猫咪 ↔ 汽车",   v100_a, v100_c)

    print("\n❌ 跨模型:相似度完全乱了(猫咪和汽车得分都能很高,无意义)")
    show("seed=42 猫咪 ↔ seed=10086 小猫 ", v42_a, v100_b)
    show("seed=42 猫咪 ↔ seed=10086 汽车 ", v42_a, v100_c)
    show("seed=42 小猫 ↔ seed=10086 小猫 ", v42_b, v100_b)  # 同一个文本都对不上!

    print("\n💡 这就是「建库和查询必须用同一个 embedding 模型」的底层原因。")
    print("   即使是同一句话,不同模型映射的位置都不一样,比相似度纯属瞎算。")


# ─────────────────────────────────────────────
# 4. 演示 ③ Embedding 和向量数据库如何配合
# ─────────────────────────────────────────────

class ToyVectorDB:
    """最简向量数据库:平行数组 + Flat 暴力 Top-K"""

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.vectors: list[list[float]] = []
        self.texts: list[str] = []
        self.metas: list[dict] = []

    def add(self, text: str, vec: list[float], meta: dict | None = None) -> None:
        """把(向量+文本+元信息)写进来。注意 vec 是外面算好塞进来的!"""
        assert len(vec) == self.dim
        self.vectors.append(vec)
        self.texts.append(text)
        self.metas.append(meta or {})

    def search(self, query_vec: list[float], k: int = 2) -> list[dict]:
        """拿查询向量,比相似度,排 Top-K"""
        scored = []
        for i in range(len(self.vectors)):
            scored.append({
                "score": cosine(query_vec, self.vectors[i]),
                "text": self.texts[i],
                "meta": self.metas[i],
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]


def demo_3_embedding_with_vectordb():
    print("\n" + "=" * 70)
    print("📌 演示 ③:【Embedding 和向量数据库是上下游关系】")
    print("=" * 70)

    SEED = 7    # 整个例子里必须用同一个 seed(同一个模型)
    db = ToyVectorDB(DIM)

    docs = [
        ("员工请假需提前 3 天 OA 申请,主管审批",      {"source": "手册.pdf", "sec": "请假"}),
        ("报销保留发票,月底前系统上传,财务 10 日打款",  {"source": "手册.pdf", "sec": "报销"}),
        ("出差一线城市酒店不超 500 元/晚",            {"source": "手册.pdf", "sec": "出差"}),
        ("会议室超过 2 小时需抄送负责人",              {"source": "办公.pdf", "sec": "会议室"}),
    ]

    # ── 建库 ──
    print("\n【建库阶段】Embedding 负责 文本→向量;向量数据库负责 存")
    for text, meta in docs:
        vec = fake_embed(text, SEED)   # ★ 上游:Embedding 算
        db.add(text, vec, meta)        # ★ 下游:写进向量库
        s = text[:20] + ("…" if len(text) > 20 else "")
        print(f"   add 「{s}」→ 向量(前 3 维: {vec[0]:+.3f}, {vec[1]:+.3f}, {vec[2]:+.3f}…)")

    # ── 查询 ──
    queries = ["怎么请假?", "发票要怎么弄?"]
    for q in queries:
        print(f"\n【查询】{q}")
        q_vec = fake_embed(q, SEED)        # ★ 上游:同一个 Embedding 模型算
        results = db.search(q_vec, k=2)    # ★ 下游:向量数据库比相似度 Top-K
        for r in results:
            print(f"   score={r['score']:.4f}  meta={r['meta']}")
            print(f"        原文: {r['text']}")

    print("\n⚠️  注意:建库时和查询时都是 fake_embed(x, seed=7) — 同一个模型!")
    print("   如果一边 seed=7 一边 seed=100,上一个演示已经证明:结果全乱。")


if __name__ == "__main__":
    demo_1_same_model_similarity()
    demo_2_different_model_space()
    demo_3_embedding_with_vectordb()
