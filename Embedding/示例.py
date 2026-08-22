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
    【这段代码是干什么的】⭐
    把一段「文字」翻译成一串「数字」(向量),让计算机能用数学方法比意思相近度。
    这就是 Embedding 模型在做的事情,只不过这里是手搓简化版。

    通俗类比:
        文字 "猫咪"  → [0.8, -0.2, 0.5, 0.1, ...]  这串数字就是"猫咪"的语义指纹
        文字 "小猫"  → [0.79, -0.21, 0.52, 0.12, ...] 数字很像 → 意思相近
        文字 "汽车"  → [-0.3, 0.9, 0.1, -0.5, ...]   数字差很远 → 意思无关

    核心思路(3 步):
        ① 给每个字符(猫/狗/汽/车...)提前随机生成一串"它的数字指纹"
           —— 同一个字符 + 同一个 seed → 永远生成同一串数字(可复现)
        ② 一段话的向量 = 把话里每个字符的指纹"对应位置相加"
           —— 字越多,累加方向越偏向那些字
        ③ L2 归一化:把向量长度拉到 1
           —— 这样后续算相似度时,不受字数多少影响,只看"方向"

    参数:
        text: 要翻译的文字
        seed: 模型种子。seed 相同 = 同一个"模型",同一个字符→同一串指纹
              seed 不同 = 不同模型,同一字符→不同指纹(空间都换了)

    真场景对应:
        真实 Embedding 模型(bge / MiniLM)是神经网络,本质上也是做这件事:
        输入文本 → 输出定长向量 → 向量方向编码语义
        只是神经网络学到了"真正"的语义,我们这里用字符哈希冒充一下。
    """
    # ① 准备一个全 0 的累加器:长度 = DIM(48),每个位置都从 0 开始
    # JS 类比: const vec = new Array(DIM).fill(0);
    vec = [0.0] * DIM

    # ② 遍历文本里每个字符,把"这个字符的指纹"累加到 vec 上
    for ch in text:
        # 用 (seed, ch) 当 key 去缓存里查:这个字符在这个模型下的指纹算过没
        key = (seed, ch)

        if key not in _CHAR_CACHE:
            # 没算过 → 第一次见到这个字符,给它生成一串随机指纹并缓存
            # 关键:用 (seed, 字符的 unicode) 做随机种子
            #   - 同一个字符 + 同一个 seed → 同一串指纹(每次跑都一样)
            #   - 换 seed → 指纹全变(等于换了个"模型")
            # random.Random(seed) = JS 里的 new Random(seed),可复现的伪随机
            rng = random.Random(hash((seed, ord(ch))) & 0xFFFFFFFF)

            # 生成 DIM 个 -1.0 ~ 1.0 之间的随机数,作为这个字符的"指纹"
            # JS 类比: _CHAR_CACHE[key] = Array.from({length: DIM}, () => rng.uniform(-1, 1));
            _CHAR_CACHE[key] = [rng.uniform(-1.0, 1.0) for _ in range(DIM)]

        # 取出这个字符的指纹(48 维的列表)
        cv = _CHAR_CACHE[key]

        # 把指纹按位加到累加器 vec 上
        # 例如:vec[i] 累加 cv[i],字符越多,vec 各维度的值越大
        # JS 类比: for (let i = 0; i < DIM; i++) vec[i] += cv[i];
        for i in range(DIM):
            vec[i] += cv[i]

    # ③ L2 归一化:让向量长度变成 1,只保留"方向"信息
    # 原因:不归一化的话,字多的文本天然模长大,相似度会偏
    # 归一化后,两个向量点积 = 夹角余弦,只看"方向像不像"
    #
    # 公式:每个分量除以向量的模长 |v| = √(v0² + v1² + ... + vD-1²)
    # JS 类比:
    #   const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0)) || 1e-9;
    #   return vec.map(v => v / norm);
    norm = math.sqrt(sum(v * v for v in vec)) or 1e-9
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """
    【这段代码是干什么的】⭐
    算两个向量"有多像":返回值范围 -1.0 ~ +1.0
        +1.0  方向完全一致 → 语义完全相同
         0    方向垂直 → 语义无关
        -1.0  方向相反 → 语义相反

    【为什么这么简单】
    正常余弦公式是:cos = (a·b) / (|a| * |b|)
    但因为 fake_embed 最后做了 L2 归一化,|a|=|b|=1,所以分母是 1,
    余弦相似度 = 直接点积 = Σ(a[i] * b[i])

    【举例】
    a = [1, 0]   b = [1, 0]   → 点积 = 1*1 + 0*0 = 1.0  (完全一样)
    a = [1, 0]   b = [0, 1]   → 点积 = 1*0 + 0*1 = 0.0  (完全无关)
    a = [1, 0]   b = [-1, 0]  → 点积 = -1.0             (完全相反)
    """
    # zip(a, b) 把两个列表"配对":[(a0,b0), (a1,b1), ...]
    # sum(ai * bi for ai, bi in zip(a, b)) = 点积 Σ(a[i]*b[i])
    # round(..., 4) 保留 4 位小数,打印好看一点
    #
    # JS 类比:
    #   function cosine(a, b) {
    #     let s = 0;
    #     for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    #     return Math.round(s * 10000) / 10000;
    #   }
    return round(sum(ai * bi for ai, bi in zip(a, b)), 4)


# ─────────────────────────────────────────────
# 1.5 单独跑个小例子:让你看清 fake_embed + cosine 到底在干什么
# ─────────────────────────────────────────────
# 直接 `python 示例.py demo_embed` 就只跑这个例子,不被后面的 3 个 demo 刷屏
# ---------------------------------------------------------------------

def demo_embed_walkthrough():
    """
    一个最小例子:把 3 个词"猫咪/小猫/汽车"都变成向量,
    然后看它们两两的相似度,直观感受 Embedding 在做什么。
    """
    print("\n" + "=" * 70)
    print("🎯 最小例子:fake_embed + cosine 在干什么")
    print("=" * 70)

    SEED = 42   # 选一个"模型"

    # ① 拿 3 个词去 Embedding
    v_cat   = fake_embed("猫咪", SEED)
    v_kitty = fake_embed("小猫", SEED)
    v_car   = fake_embed("汽车", SEED)

    # ② 看看每个词的向量长啥样(只打印前 8 维,48 维太长)
    print("\n[1] 文本 → 向量(只看前 8 维):")
    print(f"    「猫咪」 → {[round(v, 3) for v in v_cat[:8]]} … (共 {len(v_cat)} 维)")
    print(f"    「小猫」 → {[round(v, 3) for v in v_kitty[:8]]} … (共 {len(v_kitty)} 维)")
    print(f"    「汽车」 → {[round(v, 3) for v in v_car[:8]]} … (共 {len(v_car)} 维)")

    # ③ 两两算余弦相似度
    print("\n[2] 两两相似度(cosine):")
    print(f"    cos(「猫咪」, 「小猫」) = {cosine(v_cat, v_kitty):+.4f}   ← 语义近,数值高")
    print(f"    cos(「猫咪」, 「汽车」) = {cosine(v_cat, v_car):+.4f}   ← 语义远,数值低")
    print(f"    cos(「小猫」, 「汽车」) = {cosine(v_kitty, v_car):+.4f}   ← 语义远,数值低")

    # ④ 解释:为什么"猫咪 ↔ 小猫"分高?
    print("\n[3] 为什么「猫咪 ↔ 小猫」得分会高?")
    print("    因为两个词都含「猫」字,而「猫」字在我们这套手搓模型里")
    print("    生成了一串固定的指纹,两次累加都把这段指纹加进去了 → 方向接近。")
    print("    而「汽车」完全不含「猫」字,累加的是不同指纹 → 方向差远。")
    print("\n    👉 这就是 Embedding 的本质:")
    print("       让「意思像的内容」→「向量方向像」→「余弦相似度高」。")


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
    import sys
    # 用法:
    #   python 示例.py             → 跑全部 4 个 demo
    #   python 示例.py demo_embed  → 只跑最小例子,看 fake_embed + cosine 干啥
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    if arg in ("all", "demo_embed"):
        demo_embed_walkthrough()
    if arg in ("all", "demo_1"):
        demo_1_same_model_similarity()
    if arg in ("all", "demo_2"):
        demo_2_different_model_space()
    if arg in ("all", "demo_3"):
        demo_3_embedding_with_vectordb()
