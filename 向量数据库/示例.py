"""
向量数据库 · 完整可运行创建示例(零依赖纯 Python)
================================================

用纯 Python 手搓一个真正的「向量数据库」,对照知识点 Step 1-6:
  Step 1   选 Embedding 模型 & 维度 (手搓版,DIM=48)
  Step 2   准备 100 条 chunk
  Step 3   全部向量化
  Step 4   选 ToyVectorDB(相当于 FAISS/Chroma 的手搓版)
  Step 5   写入 + 建索引(两种:Flat 暴力 / IVF 倒排聚类)
  Step 6   save_local 持久化 / load_local 重新加载 + 查询验证

索引层实现:
  - FlatIndex : 全量暴力对比,100% 召回,对标 FAISS IndexFlatL2
  - IVFIndex  : 先 K-Means 分 nlist 桶,查最近 nprobe 个桶再暴力,对标 IndexIVFFlat
  - 知识点里的 HNSW 图结构手写太复杂,这里用 IVF 演示「索引加速」的原理就够了
"""

import math
import random
import json
import os
from pprint import pprint


# ─────────────────────────────────────────────
# 0. Embedding 工具(复用 Embedding 文件夹的手搓版,保持独立可运行)
# ─────────────────────────────────────────────
# 这部分代码的作用:把「文本 → 一串 48 维数字」。
# 通俗类比:给每句话拍一张「语义指纹照」,后面就靠对比指纹像不像来搜内容。
#
# 想更细的理解,看 Embedding/示例.py 里的详细讲解和 demo_embed_walkthrough。

DIM = 48                    # 向量维度:每个文本会变成长度 48 的浮点数列表
_EMB_SEED = 42              # Embedding 模型种子,固定它=固定用同一个"模型"
_CHAR_CACHE: dict[tuple, list[float]] = {}   # 字符指纹缓存,同一个字重复用不用再算


def fake_embed(text: str) -> list[float]:
    """
    【这段在干嘛】
    把一段文字翻译成 48 个浮点数组成的列表(向量),也叫「文本的 Embedding」。
    这是向量数据库建库和查询时都会调用的底层函数。

    核心 3 步(和 Embedding/示例.py 里的 fake_embed 一模一样,只是 seed 固定了):
      ① 给每个字符(猫/狗/汽/车...)准备一个固定随机数数组"指纹"
      ② 一句话的向量 = 每个字符的指纹按位累加
      ③ L2 归一化:把向量长度拉成 1,消除字数影响

    JS 类比整体:
      const CACHE = {};
      function fakeEmbed(text) {
        let vec = new Array(DIM).fill(0);
        for (const ch of text) {
          const key = `${SEED}|${ch}`;
          if (!CACHE[key]) {
            const rng = new Random(hash(SEED, ch.codePointAt(0)));
            CACHE[key] = Array.from({length: DIM}, () => rng.uniform(-1, 1));
          }
          const cv = CACHE[key];
          for (let i = 0; i < DIM; i++) vec[i] += cv[i];
        }
        const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0)) || 1e-9;
        return vec.map(v => v / norm);
      }
    """
    # ① 准备一个 48 位全 0 的累加器(装整段文本的「合成指纹」)
    vec = [0.0] * DIM

    for ch in text:
        key = (_EMB_SEED, ch)                      # (模型,字符)当 key
        if key not in _CHAR_CACHE:                  # 没算过这个字符 → 第一次生成
            # 用 (模型seed + 字符unicode) 做随机数种子 → 同一个字符永远生成同样指纹
            rng = random.Random(hash((_EMB_SEED, ord(ch))) & 0xFFFFFFFF)
            # 生成 48 个 -1~1 的随机数,作为这个字符的 48 维指纹
            _CHAR_CACHE[key] = [rng.uniform(-1.0, 1.0) for _ in range(DIM)]
        cv = _CHAR_CACHE[key]                       # 取出该字符的指纹
        for i in range(DIM):                        # 把字符指纹累加到整段文本的合成指纹上
            vec[i] += cv[i]
    # ③ L2 归一化:让合成指纹长度=1,后续相似度只看"方向"不看字数
    norm = math.sqrt(sum(v * v for v in vec)) or 1e-9
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """
    【这段在干嘛】
    算两个向量"有多像"。因为 fake_embed 做了 L2 归一化(|a|=|b|=1),
    余弦公式 cos = (a·b)/(|a|·|b|) 里的分母是 1,所以直接做「点积 Σa[i]*b[i]」就行。

    返回范围: -1.0 ~ +1.0
      +1  方向完全一致 → 意思基本一样
       0  方向垂直     → 意思无关
      -1  方向相反     → 意思相反

    JS 类比:
      function cosine(a, b) {
        let s = 0;
        for (let i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;   // 保留 4 位小数交给调用方自己做
      }
    """
    # zip(a, b) 把两个列表"对齐配对":[(a0,b0), (a1,b1), ...]
    # 每对相乘,然后加起来 = 点积
    return sum(ai * bi for ai, bi in zip(a, b))


# ─────────────────────────────────────────────
# 1. 索引层抽象:FlatIndex(暴力) / IVFIndex(倒排聚类加速)
# ─────────────────────────────────────────────
# 这是向量数据库内部「加速近邻检索」的核心部分。
# 真实产品有 HNSW/DiskANN/PQ 等,这里用最容易实现的 IVF 演示原理。

class FlatIndex:
    """
    Flat 暴力索引:对所有向量逐个比。
    对标 FAISS:IndexFlatL2 / IndexFlatIP
    数据量 ≤ 1 万时 100% 召回,速度也够用。
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors: list[list[float]] = []  # 所有向量

    def add(self, vectors: list[list[float]]) -> None:
        for v in vectors:
            assert len(v) == self.dim, f"维度不匹配,期望 {self.dim},实际 {len(v)}"
        self.vectors.extend(vectors)

    def search(self, q_vec: list[float], k: int) -> tuple[list[int], list[float]]:
        """
        返回 (top_k 的位置下标列表, 对应 score 列表),均按 score 从大到小排序。
        """
        scored = [(i, cosine(q_vec, self.vectors[i])) for i in range(len(self.vectors))]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:k]
        return [x[0] for x in top], [round(x[1], 4) for x in top]

    # ---- 序列化(为了 save_local) ----
    def state_dict(self) -> dict:
        return {"type": "flat", "dim": self.dim, "vectors": self.vectors}

    @classmethod
    def from_state(cls, state: dict) -> "FlatIndex":
        idx = cls(state["dim"])
        idx.vectors = state["vectors"]
        return idx


class IVFIndex:
    """
    IVF 倒排索引(Inverted File Index):
    建库阶段:
        1. 对所有向量做 K-Means 聚类,得到 nlist 个「聚类中心(centroid)」
        2. 每条向量归到最近的 1 个聚类桶里(倒排)
    查询阶段:
        1. 先找 q_vec 最近的 nprobe 个聚类中心
        2. 只在这 nprobe 个桶里做暴力
        3. 取 Top-K

    对标 FAISS:IndexIVFFlat
    典型参数:nlist = sqrt(N) 左右;  nprobe = 10~50 (越大越慢,召回越高)
    """

    def __init__(self, dim: int, nlist: int = 16, nprobe: int = 4, seed: int = 7):
        self.dim = dim
        self.nlist = nlist         # 聚类中心数 = 桶数
        self.nprobe = nprobe       # 查询时扫多少个桶
        self.seed = seed
        self.centroids: list[list[float]] = []   # 聚类中心 (nlist, dim)
        self.buckets: list[list[int]] = []       # buckets[cid] = 该桶里的 row id 列表

    # ── 建库:K-Means 聚类 + 建倒排桶 ──
    def add(self, vectors: list[list[float]], max_iter: int = 20) -> None:
        N = len(vectors)
        self.nlist = min(self.nlist, N)   # 桶数不能超点数
        rng = random.Random(self.seed)

        # 1) 初始化中心:随机挑 nlist 条向量
        picks = rng.sample(range(N), self.nlist)
        self.centroids = [list(vectors[i]) for i in picks]

        assign = [0] * N          # assign[row] = 这条归到哪个桶
        for it in range(max_iter):
            # 2a) E 步:每条向量找最近的中心
            changed = False
            for i in range(N):
                best_c, best_s = 0, -1.0
                for c in range(self.nlist):
                    s = cosine(vectors[i], self.centroids[c])
                    if s > best_s:
                        best_s = s
                        best_c = c
                if assign[i] != best_c:
                    assign[i] = best_c
                    changed = True
            # 2b) M 步:每个桶内的向量求均值,更新为新中心
            for c in range(self.nlist):
                mems = [i for i in range(N) if assign[i] == c]
                if not mems:
                    # 空桶:重新随机给一个向量
                    self.centroids[c] = list(vectors[rng.randrange(N)])
                    continue
                new_c = [0.0] * self.dim
                for mi in mems:
                    for d in range(self.dim):
                        new_c[d] += vectors[mi][d]
                for d in range(self.dim):
                    new_c[d] /= len(mems)
                # 重新归一化(向量空间是单位球面)
                norm = math.sqrt(sum(v * v for v in new_c)) or 1e-9
                self.centroids[c] = [v / norm for v in new_c]
            if not changed:
                break

        # 3) 最终生成倒排桶
        self.buckets = [[] for _ in range(self.nlist)]
        for i in range(N):
            self.buckets[assign[i]].append(i)

    # ── 查询:先找 nprobe 个最近桶 → 只在这些桶里暴力 ──
    def search(self, q_vec: list[float], k: int) -> tuple[list[int], list[float]]:
        # 1) 给 q_vec 找最近的 nprobe 个中心
        c_scores = [(c, cosine(q_vec, self.centroids[c])) for c in range(self.nlist)]
        c_scores.sort(key=lambda x: x[1], reverse=True)
        probe_cids = [x[0] for x in c_scores[: self.nprobe]]

        # 2) 把这些桶里的 row id 全部收集起来(去重)
        cand_rows = set()
        for cid in probe_cids:
            for rid in self.buckets[cid]:
                cand_rows.add(rid)

        # 3) 在候选集里暴力比相似度,取 Top-K
        scored = [(r, cosine(q_vec, _GLOBAL_VECTORS[r])) for r in cand_rows]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:k]
        return [x[0] for x in top], [round(x[1], 4) for x in top]

    # ── 序列化 ──
    def state_dict(self) -> dict:
        return {
            "type": "ivf",
            "dim": self.dim,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "centroids": self.centroids,
            "buckets": self.buckets,
        }

    @classmethod
    def from_state(cls, state: dict) -> "IVFIndex":
        idx = cls(state["dim"], state["nlist"], state["nprobe"])
        idx.centroids = state["centroids"]
        idx.buckets = state["buckets"]
        return idx


# 这是一个全局小技巧:IVFIndex.search 需要引用所有向量做相似度
# 实际 FAISS 是在 C 内部共享,Python 手搓版我们用一个模块级变量传进去。
_GLOBAL_VECTORS: list[list[float]] = []


# ─────────────────────────────────────────────
# 2. ToyVectorDB:数据库层(数据层 + 索引层 + 服务层三合一)
# ─────────────────────────────────────────────
# 知识点 6 步流程在这里完整走完:
#   Step 1:__init__ 确定 DIM
#   Step 2:add_texts 外部已切好 chunk
#   Step 3:add_texts 内部调 fake_embed 完成向量化
#   Step 4:用 ToyVectorDB
#   Step 5:build_index 选 flat/ivf 建索引
#   Step 6:save_local / load_local 持久化,search 查询

class ToyVectorDB:
    """真正的「向量数据库」:把数据存起来+索引加速+CRUD+持久化"""

    def __init__(self, dim: int = DIM, embedding_model_name: str = "fake-embed-v1"):
        self.dim = dim
        self.embedding_model_name = embedding_model_name  # ★ 记录模型名,防混用

        # Layer 1: 数据层
        self.row_ids: list[str] = []          # 用户给的 id,对应行号
        self.vectors: list[list[float]] = []  # (N, dim) 核心向量数组
        self.texts: list[str] = []
        self.metas: list[dict] = []

        # Layer 2: 索引层(一开始 None,调用 build_index 才建)
        self.index: FlatIndex | IVFIndex | None = None
        self.index_type: str = "none"

    # ── Step 2 + Step 3:写入 chunk,内部同步做 Embedding ──
    def add_texts(self, id_text_meta: list[tuple[str, str, dict | None]]) -> None:
        """批量添加:[(id, text, meta), ...]"""
        start_row = len(self.vectors)
        for row, (id_, text, meta) in enumerate(id_text_meta, start=start_row):
            self.row_ids.append(id_)
            vec = fake_embed(text)
            self.vectors.append(vec)
            self.texts.append(text)
            self.metas.append(meta or {})

        # 索引如果已经建了,追加新向量也要进索引(我们简单起见:建一次就定了,
        # 真实产品 FAISS/Milvus 支持 add 追加)
        if self.index is not None:
            self.index.add(self.vectors[start_row:])

    # ── Step 5:建索引(把「知识点 IVF / Flat」选一个) ──
    def build_index(self, index_type: str = "flat", **kwargs) -> None:
        """
        index_type ∈ {flat, ivf}
        ivf 的参数:nlist(桶数), nprobe(查询扫桶数)
        """
        global _GLOBAL_VECTORS
        _GLOBAL_VECTORS = self.vectors  # 给 IVF 索引查询时用

        if index_type == "flat":
            idx = FlatIndex(self.dim)
            idx.add(self.vectors)
        elif index_type == "ivf":
            idx = IVFIndex(
                self.dim,
                nlist=kwargs.get("nlist", 16),
                nprobe=kwargs.get("nprobe", 4),
            )
            idx.add(self.vectors)
        else:
            raise ValueError(f"未知 index_type: {index_type},可选 flat/ivf")

        self.index = idx
        self.index_type = index_type

    # ── 查询:返回 Top-K 结构化结果 ──
    def search(self, query: str, k: int = 3, query_vec: list[float] | None = None) -> list[dict]:
        """
        传入 query 文本或直接传 query_vec(做索引对比时省 CPU)。
        返回 [{"id","text","meta","score","row"}, ...]
        """
        if self.index is None:
            raise RuntimeError("还没建索引,请先调用 build_index('flat'/'ivf')")

        qv = query_vec if query_vec is not None else fake_embed(query)
        rows, scores = self.index.search(qv, k)
        out = []
        for row, s in zip(rows, scores):
            out.append({
                "row": row,
                "id": self.row_ids[row],
                "score": s,
                "text": self.texts[row],
                "meta": self.metas[row],
            })
        return out

    # ── 持久化 save + load (对应知识点磁盘存储) ──
    # 真实产品 Chroma/FAISS 会拆成几个文件,我们打包成 2 个文件方便看:
    #   <dir>/db_meta.json  ← id/texts/metas + 头信息
    #   <dir>/index.json    ← 索引的 vectors/centroids/buckets

    def save_local(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        meta_path = os.path.join(directory, "db_meta.json")
        index_path = os.path.join(directory, "index.json")

        # 数据层
        db_meta = {
            "dim": self.dim,
            "embedding_model_name": self.embedding_model_name,
            "index_type": self.index_type,
            "n_rows": len(self.vectors),
            "row_ids": self.row_ids,
            "vectors": self.vectors,    # ★ 数据层核心:把向量数组也放进 meta 文件
            "texts": self.texts,
            "metas": self.metas,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(db_meta, f, ensure_ascii=False, indent=2)

        # 索引层
        with open(index_path, "w", encoding="utf-8") as f:
            state = self.index.state_dict() if self.index else {"type": "none"}
            json.dump(state, f, ensure_ascii=False, indent=2)

        print(f"💾 向量库已持久化到磁盘: {os.path.abspath(directory)}")
        print(f"   📄 {meta_path}  ({os.path.getsize(meta_path)//1024} KB)")
        print(f"   📄 {index_path} ({os.path.getsize(index_path)//1024} KB)")

    @classmethod
    def load_local(cls, directory: str, expected_embedding_model: str | None = None) -> "ToyVectorDB":
        """
        从磁盘重新加载。
        expected_embedding_model:如果传了,和库记录的模型名不一致会抛异常
                                (知识点坑 1:建库和查询必须同模型!)
        """
        meta_path = os.path.join(directory, "db_meta.json")
        index_path = os.path.join(directory, "index.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            db_meta = json.load(f)

        if expected_embedding_model and db_meta["embedding_model_name"] != expected_embedding_model:
            raise ValueError(
                f"Embedding 模型不匹配!库是 {db_meta['embedding_model_name']}, "
                f"你给的是 {expected_embedding_model},请删库重建。"
            )

        db = cls(db_meta["dim"], db_meta["embedding_model_name"])
        db.row_ids = db_meta["row_ids"]
        db.vectors = db_meta["vectors"]   # ★ 从 db_meta 读,不再依赖索引文件
        db.texts = db_meta["texts"]
        db.metas = db_meta["metas"]
        db.index_type = db_meta["index_type"]

        # 索引层:根据 type 还原(只存 centroids/buckets,不重复存 vectors)
        global _GLOBAL_VECTORS
        with open(index_path, "r", encoding="utf-8") as f:
            idx_state = json.load(f)
        if idx_state["type"] == "flat":
            # Flat 索引的 state 里有 vectors(和 db.vectors 一样)
            db.index = FlatIndex.from_state(idx_state)
        elif idx_state["type"] == "ivf":
            # IVF 索引只存 centroids + buckets,vectors 通过全局共享 db.vectors
            db.index = IVFIndex.from_state(idx_state)
        else:
            raise RuntimeError(f"load_local:未知索引类型 {idx_state['type']}")

        _GLOBAL_VECTORS = db.vectors
        print(f"📂 已从磁盘加载向量库:{os.path.abspath(directory)}")
        print(f"   共 {len(db.vectors)} 条,维度 {db.dim},索引 {db.index_type}")
        return db

    # ── inspect:看内部长啥样 ──
    def inspect(self) -> None:
        print("\n" + "=" * 70)
        print(f"🧮 ToyVectorDB 内部快照 | dim={self.dim} | N={len(self.vectors)} | "
              f"索引={self.index_type} | 模型={self.embedding_model_name}")
        print("=" * 70)
        show_n = min(6, len(self.vectors))
        for i in range(show_n):
            vec_head = ", ".join(f"{v:+.3f}" for v in self.vectors[i][:4])
            text_s = self.texts[i][:26] + ("…" if len(self.texts[i]) > 26 else "")
            print(f"  [{i:>3}] id={self.row_ids[i]}")
            print(f"        vec  : [{vec_head}, …] ({len(self.vectors[i])} 维)")
            print(f"        text : {text_s}")
            print(f"        meta : {self.metas[i]}")
        if self.index_type == "ivf":
            assert isinstance(self.index, IVFIndex)
            lens = sorted([len(b) for b in self.index.buckets])
            print(f"\n   🪣 IVF 桶大小:min={min(lens)}, median={lens[len(lens)//2]},"
                  f" max={max(lens)}, 桶数={len(self.index.buckets)}")
        print("=" * 70)


# ─────────────────────────────────────────────
# 3. 造一批「模拟 chunk 数据」(100 条,多个主题混合)
# ─────────────────────────────────────────────

def build_mock_chunks(n_per_topic: int = 20) -> list[tuple[str, str, dict]]:
    """
    6 个主题,每个主题 n_per_topic 条,共 6*20=120 条,
    用于测试 IVF 是否能按「主题聚桶」和查询召回率。
    """
    topics = [
        ("LEAVE",  "请假制度", "员工手册.pdf",     "年假全年 10 天,病假凭医院证明销假,需提前 {n} 天在 OA 提交请假申请,经部门主管审批后生效。"),
        ("EXP",    "报销制度", "员工手册.pdf",     "因公消费需保留发票,当月月底前在报销系统上传电子发票,报销上限 {n} 元,财务 10 个工作日内打款到工资卡。"),
        ("TRIP",   "出差制度", "员工手册.pdf",     "一线城市差旅住宿不超过 {n} 元每晚,高铁二等座,飞机经济舱,超过标准需上级特批,出差前提交差旅申请。"),
        ("MEET",   "会议室",   "办公指南.pdf",     "通过内部系统预订会议室,时长超过 {n} 小时需抄送部门负责人,会议结束请清理白板。"),
        ("HR",     "工资薪酬", "员工手册.pdf",     "每月 {n} 日发放上月工资,逢节假日顺延至下一工作日,工资条在 HR 系统查看,年终奖按绩效 A/B/C/D 评定。"),
        ("CHECK",  "打卡出勤", "行政制度.pdf",     "工作时间 9:00-18:00,打卡时间早 9 晚 6,当月累计迟到 {n} 次以上扣绩效分 5 分,年假可冲抵迟到。"),
    ]
    rng = random.Random(123)
    result = []
    for tag, section, source, tmpl in topics:
        for i in range(n_per_topic):
            # 把模板里的 {n} 换成不同数字,让每条文本不重复但语义相近
            number = rng.randint(1, 30) if tag != "HR" else rng.randint(10, 20)
            text = tmpl.format(n=number)
            id_ = f"{tag}_{i:03d}"
            meta = {"source": source, "section": section, "tag": tag}
            result.append((id_, text, meta))
    rng.shuffle(result)   # 打乱顺序,看 IVF 能不能自动聚回同类
    return result


# ─────────────────────────────────────────────
# 4. 主流程:演示创建→建索引→save→load→查询 全流程
# ─────────────────────────────────────────────

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_vector_db")

def demo_full_workflow():
    print("\n" + "#" * 70)
    print("# 🎯 演示:从零创建一个向量数据库(对应知识点 Step 1-6)")
    print("#" * 70)

    # Step 1:确定模型和维度(ToyVectorDB __init__ 已固定 DIM=48)
    # Step 2:准备 chunks
    chunks = build_mock_chunks(n_per_topic=20)
    print(f"\nStep 2 ✅ 准备好 {len(chunks)} 条 chunk,6 个主题各 20 条")

    db = ToyVectorDB(DIM, embedding_model_name="fake-embed-v1")
    # Step 3:向量化 + 写入
    db.add_texts(chunks)
    print(f"Step 3 ✅ 已全部向量化并写入:共 {len(db.vectors)} 条 × {db.dim} 维")

    # Step 4:ToyVectorDB 本身就是「数据库产品」
    # Step 5:建索引(先建 Flat 做基线;再建 IVF 看加速)
    db.build_index("ivf", nlist=12, nprobe=3)
    print(f"Step 5 ✅ 建 IVF 索引:nlist=12 桶,nprobe=3 查询桶")
    db.inspect()

    # Step 6 part 1:save_local 持久化
    db.save_local(SAVE_DIR)

    # 模拟关进程:删掉 db
    del db
    print("\n♻️  已删除内存对象 db,模拟「程序关闭」")

    # Step 6 part 2:load_local 重新加载
    print("\n📂 从磁盘重新加载...")
    db2 = ToyVectorDB.load_local(SAVE_DIR, expected_embedding_model="fake-embed-v1")

    # 用错误模型试试(验证知识点坑 1)
    print("\n🔒 试试传入错误 embedding 模型名(预期报错):")
    try:
        ToyVectorDB.load_local(SAVE_DIR, expected_embedding_model="bge-large-zh")
    except ValueError as e:
        print(f"   ✅ 拦截成功,防混用模型: {e}")

    # Step 6 part 3:查几个问题
    queries = [
        "请假怎么申请?",
        "发票报销流程?",
        "什么时候发工资?",
        "订超过 2 小时的会议室",
    ]
    for q in queries:
        print(f"\n💬 查询: {q}")
        results = db2.search(q, k=2)
        for r in results:
            s = r["text"][:40] + ("…" if len(r["text"]) > 40 else "")
            print(f"   score={r['score']:.4f}  id={r['id']:<10} tag={r['meta']['tag']}")
            print(f"         {s}")


def demo_compare_flat_vs_ivf():
    """演示两种索引的召回差异:Flat 100%,IVF 近似"""
    print("\n" + "#" * 70)
    print("# ⚖️  对比:Flat 索引(暴力 100% 召回) vs IVF 索引(近似加速)")
    print("#" * 70)

    chunks = build_mock_chunks(n_per_topic=25)   # 150 条
    # 先做 20 条随机查询,看 IVF 在前 3 名里,和 Flat 对得上几条
    rng = random.Random(42)
    random_queries = [c[1] for c in rng.sample(chunks, 20)]

    # Flat
    db_flat = ToyVectorDB(embedding_model_name="cmp-v1")
    db_flat.add_texts(chunks)
    db_flat.build_index("flat")

    # IVF:桶少一点,召回会下降,更能看出区别
    db_ivf = ToyVectorDB(embedding_model_name="cmp-v1")
    db_ivf.add_texts(chunks)
    db_ivf.build_index("ivf", nlist=6, nprobe=1)

    hit_top1 = 0
    hit_top3_in_top5 = 0
    for q in random_queries:
        qv = fake_embed(q)
        flat = db_flat.search("", k=1, query_vec=qv)
        ivf = db_ivf.search("", k=5, query_vec=qv)
        if flat[0]["row"] == ivf[0]["row"]:
            hit_top1 += 1
        flat_top3_ids = {r["row"] for r in db_flat.search("", k=3, query_vec=qv)}
        ivf_top5_ids = {r["row"] for r in ivf}
        if len(flat_top3_ids & ivf_top5_ids) >= 3:
            hit_top3_in_top5 += 1

    total = len(random_queries)
    print(f"\n共 {total} 条随机查询:")
    print(f"   Top-1 完全命中(Flat=IVF 第 1 名):{hit_top1:>3} / {total}  =  {hit_top1/total*100:5.1f}%")
    print(f"   Flat Top-3 ⊆ IVF Top-5 (召回): {hit_top3_in_top5:>3} / {total}  =  {hit_top3_in_top5/total*100:5.1f}%")
    print("\n💡 把 nprobe 从 1 调到 4 再重跑,召回会明显提升;这就是 IVF 的「速度 ↔ 召回率」旋钮。")


def show_disk_files():
    """把持久化的磁盘文件打开给你看一眼"""
    print("\n" + "#" * 70)
    print("# 💾 看看 save_local 写出去的文件长啥样(知识点 Layer 1)")
    print("#" * 70)
    for fn in ["db_meta.json", "index.json"]:
        path = os.path.join(SAVE_DIR, fn)
        if not os.path.exists(path):
            continue
        print(f"\n📄 {path} (前 500 字节预览):")
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        print(txt[:500])
        if len(txt) > 500:
            print(f"\n   ……(省略 {len(txt) - 500} 字节)")


if __name__ == "__main__":
    demo_full_workflow()
    demo_compare_flat_vs_ivf()
    show_disk_files()
