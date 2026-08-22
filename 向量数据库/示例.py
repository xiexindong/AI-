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
# 1. 索引层:FlatIndex(暴力 100% 召回) / IVFIndex(倒排聚类加速)
# ─────────────────────────────────────────────
#
# 【索引层到底是干嘛的】⭐
# 数据只有几十条时,查询一条条比没问题(O(N))。
# 但到 100 万条 × 1024 维时,每次查询要算 10 亿次乘法 → 慢到离谱。
# 索引层的作用就是:以"丢失一点点召回率"为代价,把复杂度从 O(N) 降到 O(log N)。
#
# 通俗类比:
#   Flat 暴力索引 = 想找一本书,把图书馆 1 万本书从头到尾翻一遍
#   IVF 聚类索引 = 先把书按主题分到 100 个书架(K-Means聚类),想找书先去最近的3个书架找
#   HNSW 分层图   = 图书馆每层楼都标了"这层是啥",你 3 楼→2 楼→1 楼跳着找(最快)
#
# 真实向量数据库(FAISS/Milvus/Qdrant)默认基本都用 HNSW,但 HNSW 图手搓太复杂,
# 这里用 IVF(原理最容易讲清楚)来演示「索引怎么加速」。

class FlatIndex:
    """
    【Flat 暴力索引】
    不做任何加速,把库里所有向量和查询向量挨个比相似度,排序取前 K。
    好处是 100% 不会漏(召回率 100%),坏处是慢。

    对标真实产品:FAISS IndexFlatIP / Chroma 小数据量默认行为
    适用场景:≤1 万条数据,或做「标准答案池」测其他索引召回率时用。
    """

    def __init__(self, dim: int):
        self.dim = dim                          # 维度
        self.vectors: list[list[float]] = []    # 所有向量直接放在一个大列表里

    def add(self, vectors: list[list[float]]) -> None:
        """把一批向量写进索引,维度不对会立即报错防止污染"""
        for v in vectors:
            assert len(v) == self.dim, f"维度不匹配,期望 {self.dim},实际 {len(v)}"
        self.vectors.extend(vectors)

    def search(self, q_vec: list[float], k: int) -> tuple[list[int], list[float]]:
        """
        暴力 Top-K 查询。
        返回值:
          ([row_1, row_2, ..., row_k],  ← 相似向量在 self.vectors 里的位置下标
           [sco_1, sco_2, ..., sco_k]) ← 对应的相似度(从大到小排好)
        """
        # 枚举每条向量,记录 (位置下标, 相似度)
        scored = [(i, cosine(q_vec, self.vectors[i])) for i in range(len(self.vectors))]
        # 按相似度从大到小排序
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:k]  # 取前 K 个
        # 把 [(idx, sco), ...] 拆成 两个平行列表
        return [x[0] for x in top], [round(x[1], 4) for x in top]

    # ---- state_dict / from_state:序列化 → 给 save_local 存磁盘用 ----
    # 把对象里的字段打包成可 JSON 化的 dict,反过来从 dict 恢复对象
    # JS 类比: { ...obj } ↔ Object.assign(new X(), state)
    def state_dict(self) -> dict:
        return {"type": "flat", "dim": self.dim, "vectors": self.vectors}

    @classmethod
    def from_state(cls, state: dict) -> "FlatIndex":
        idx = cls(state["dim"])
        idx.vectors = state["vectors"]
        return idx


class IVFIndex:
    """
    【IVF 倒排索引:Inverted File Index】⭐ 这是本文件最复杂的一段
    思路 = 「垃圾分类加速查找」:
      建库时用 K-Means 把所有向量分成 nlist 个「桶」(就像把 10000 本书分到 100 个主题书架)
      查询时只在最近的 nprobe 个桶里做暴力(找书先去 3 个最相关的书架找,不是全图书馆)

    代价:万一有本书分错桶,或者查询向量刚好靠在两个桶边界,可能漏掉正确结果(召回率 96~99%)。
        调大 nprobe(多扫几个桶),召回率就升,速度就降,nprobe 是「速度↔召回」的旋钮。

    对标产品:FAISS IndexIVFFlat
    典型经验值: nlist ≈ sqrt(N) (sqrt(100 万)=1000),  nprobe = 10~50
    """

    def __init__(self, dim: int, nlist: int = 16, nprobe: int = 4, seed: int = 7):
        self.dim = dim
        self.nlist = nlist                    # 聚类中心数 = 「书架数」
        self.nprobe = nprobe                  # 查询时扫几个书架
        self.seed = seed                      # K-Means 初始化随机种子(保证可复现)
        self.centroids: list[list[float]] = []   # 每个书架的"代表向量"(聚类中心)
        self.buckets: list[list[int]] = []       # 每个书架里的书(row 编号列表)

    # ── 建库阶段:K-Means 聚类算法,把所有向量分桶 ──
    def add(self, vectors: list[list[float]], max_iter: int = 20) -> None:
        """
        把 vectors 做 K-Means 聚类:
          ① 初始化 nlist 个中心(随机挑 nlist 条向量当书架代表)
          ② 迭代 E 步(每条选最近的中心) + M 步(重新计算每个书架的新代表)
          ③ 直到没人换书架(收敛)或超 max_iter 轮

        通俗类比(给 10 个球分 3 个桶):
          先随便挑 3 个球当"代表"
          循环:(把每个球放到离它最近的代表那桶) → (每桶中心重新计算为桶里球的平均位置)
          直到连续一轮没人换桶,结束。
        """
        N = len(vectors)
        self.nlist = min(self.nlist, N)   # 点数比桶数少,就缩桶数
        rng = random.Random(self.seed)

        # 1) 初始化 nlist 个聚类中心:从向量里随机挑 nlist 条作为"初始书架代表"
        picks = rng.sample(range(N), self.nlist)           # 随机抽 nlist 个不同下标
        self.centroids = [list(vectors[i]) for i in picks]  # 深拷贝(怕原数据被改)

        # assign[row] = 这条向量现在被分到第几桶(长度 N 的数组)
        assign = [0] * N

        # 2) 迭代最多 max_iter 次 K-Means: E步(分配) → M步(更新中心)
        for it in range(max_iter):
            # ─────────────────── E 步:每条向量找最近的中心(给它分桶) ───────────────────
            changed = False
            for i in range(N):
                best_c, best_s = 0, -1.0
                for c in range(self.nlist):
                    # 这条向量 跟 第 c 个中心 比相似度
                    s = cosine(vectors[i], self.centroids[c])
                    if s > best_s:
                        best_s = s
                        best_c = c
                if assign[i] != best_c:    # 如果桶号变了,记一下这轮有变化
                    assign[i] = best_c
                    changed = True

            # ─────────────────── M 步:每个桶重新选代表(桶里向量取均值) ───────────────────
            for c in range(self.nlist):
                mems = [i for i in range(N) if assign[i] == c]   # 这桶里有哪些成员
                if not mems:
                    # 空桶(没人选这个中心):随机塞一条向量当代表,避免中心位置消失
                    self.centroids[c] = list(vectors[rng.randrange(N)])
                    continue
                # 新中心 = 所有成员向量按维求平均
                new_c = [0.0] * self.dim
                for mi in mems:
                    for d in range(self.dim):
                        new_c[d] += vectors[mi][d]
                for d in range(self.dim):
                    new_c[d] /= len(mems)
                # 求完均值再做一次 L2 归一化(向量在单位球面上,均值点不一定在单位球面上)
                norm = math.sqrt(sum(v * v for v in new_c)) or 1e-9
                self.centroids[c] = [v / norm for v in new_c]

            # 这一轮没人换桶 → 收敛了,不用继续迭代
            if not changed:
                break

        # 3) 收敛完成:最终生成倒排桶(每个桶里有哪些 row 编号,方便查的时候直接拿)
        self.buckets = [[] for _ in range(self.nlist)]
        for i in range(N):
            self.buckets[assign[i]].append(i)

    # ── 查询阶段:找 nprobe 个最近桶 → 只在这些桶里暴力 ──
    def search(self, q_vec: list[float], k: int) -> tuple[list[int], list[float]]:
        """
        3 步加速查询(比 Flat 省了「1 - nprobe/nlist」的计算量):
          ① 给查询向量找最近的 nprobe 个书架(聚类中心)
          ② 把这些书架里所有 row 编号合并去重 → 候选项(少了 10 倍~100 倍)
          ③ 只对候选项做暴力比相似度,排 Top-K
        """
        # 1) q_vec 和每个书架代表比相似度,取最近的 nprobe 个书架编号
        c_scores = [(c, cosine(q_vec, self.centroids[c])) for c in range(self.nlist)]
        c_scores.sort(key=lambda x: x[1], reverse=True)
        probe_cids = [x[0] for x in c_scores[: self.nprobe]]

        # 2) 把这几个书架里的所有书(row号)合并成候选集(set自动去重)
        cand_rows = set()
        for cid in probe_cids:
            for rid in self.buckets[cid]:
                cand_rows.add(rid)

        # 3) 只在候选集里做暴力,取 Top-K(用 _GLOBAL_VECTORS 拿到全局原始向量)
        scored = [(r, cosine(q_vec, _GLOBAL_VECTORS[r])) for r in cand_rows]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:k]
        return [x[0] for x in top], [round(x[1], 4) for x in top]

    # ── 序列化:存 centroids 和 buckets(vectors 已经在 ToyVectorDB 里存一份,不重复存) ──
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


# 这是一个全局共享小技巧:
# IVFIndex.search 里要拿到"完整的向量大数组"才能算 cosine。
# 真实 FAISS 在 C 内部就共享了数据,Python 手搓版我们用一个模块级变量传进去。
# ToyVectorDB.build_index 时会把 self.vectors 赋给这个变量。
_GLOBAL_VECTORS: list[list[float]] = []


# ─────────────────────────────────────────────
# 2. ToyVectorDB:数据层 + 索引层 + 服务层,三合一 = 一个真正的「向量数据库」
# ─────────────────────────────────────────────
#
# 知识点 6 步创建流程就在这里落地:
#   Step 1  确定维度/模型                →  __init__(dim=48, embedding_model_name=...)
#   Step 2  外部准备好切分 chunk          →  作为 add_texts 的入参传进来
#   Step 3  chunk → 向量 + 写入           →  add_texts(内部调 fake_embed + 存进4个平行列表)
#   Step 4  ToyVectorDB 本身就是数据库产品
#   Step 5  选 Flat/IVF 建索引           →  build_index("flat" / "ivf")
#   Step 6  存磁盘 / 读磁盘 / 查询        →  save_local / load_local / search
#
# 和真实产品的 API 对齐:
#   ToyVectorDB.add_texts   ≈  Chroma.from_texts / FAISS index.add(xb)
#   ToyVectorDB.build_index ≈  FAISS IndexIVFFlat.train(vectors)
#   ToyVectorDB.search      ≈  db.similarity_search(query, k=3)
#   ToyVectorDB.save_local  ≈  db.persist() / faiss.write_index()
#   ToyVectorDB.load_local  ≈  Chroma(persist_dir=...) / faiss.read_index()

class ToyVectorDB:
    """
    【这是整个文件的主类】
    内部其实就是 4 个平行的 Python 列表(数据层) + 一个索引对象(索引层):

      self.row_ids  = [ "ch_001",   "ch_002",   ... ]  ← 每条记录的用户ID
      self.vectors  = [ [0.1,...], [0.3,...], ... ]  ← 每条的向量(核心!)
      self.texts    = [ "请假流程", "报销流程",  ... ]  ← 原始文本
      self.metas    = [ {page:8},  {page:15},  ... ]  ← metadata(来源、页码等)
      self.index    = FlatIndex() 或 IVFIndex()      ← 加速检索的索引
    """

    def __init__(self, dim: int = DIM, embedding_model_name: str = "fake-embed-v1"):
        self.dim = dim
        # ★ 记录建库时用的模型名,以后加载时校验,防止「建库 bge,查询 MiniLM」
        self.embedding_model_name = embedding_model_name

        # Layer 1 数据层:4 个平行数组,相同位置下标对应同一条记录(行号 row)
        self.row_ids: list[str] = []
        self.vectors: list[list[float]] = []
        self.texts: list[str] = []
        self.metas: list[dict] = []

        # Layer 2 索引层:一开始没建索引是 None,必须先调 build_index 才能 search
        self.index: FlatIndex | IVFIndex | None = None
        self.index_type: str = "none"

    # ─────────────────── Step 2 + Step 3:批量添加 chunk ───────────────────
    def add_texts(self, id_text_meta: list[tuple[str, str, dict | None]]) -> None:
        """
        把 [(id, text, meta), (id, text, meta), ...] 一批 chunk 写进数据库。
        同步做 Step 3:每条都调 fake_embed 转成向量。

        真场景等价:
          texts = ["员工请假...", "报销制度..."]
          ids   = ["ch_001", "ch_002"]
          metas = [{page:8}, {page:15}]
          # Chroma 会自动做 embedding:
          db = Chroma.from_texts(texts=texts, ids=ids, metadatas=metas, embedding=emb)
        """
        # start_row:写之前有多少条,新的 chunk 就从这行号开始续
        start_row = len(self.vectors)
        # enumerate(..., start=start_row) 让 row 编号接在已有条目后面
        for row, (id_, text, meta) in enumerate(id_text_meta, start=start_row):
            self.row_ids.append(id_)                              # 写 id
            vec = fake_embed(text)                                 # 做 Embedding
            self.vectors.append(vec)                              # 写向量
            self.texts.append(text)                               # 写原文本
            self.metas.append(meta or {})                         # 写 metadata(空的话填 {})

        # 如果索引已经建好了,追加的向量也得塞进去(真实 FAISS/Milvus 支持 add 追加)
        if self.index is not None:
            self.index.add(self.vectors[start_row:])

    # ─────────────────── Step 5:建索引 ───────────────────
    def build_index(self, index_type: str = "flat", **kwargs) -> None:
        """
        选索引类型并构建。
        index_type ∈ {"flat", "ivf"}
        kwargs 接受:
          ivf 用: nlist(桶数,默认16) / nprobe(查询扫桶数,默认4)
        """
        global _GLOBAL_VECTORS
        _GLOBAL_VECTORS = self.vectors   # 给 IVFIndex.search 用(避免传给索引对象的 copies)

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

    # ─────────────────── 查询 Top-K ───────────────────
    def search(self, query: str, k: int = 3, query_vec: list[float] | None = None) -> list[dict]:
        """
        查和 query 最像的前 k 条记录,返回结构化字典列表。
        可以直接传 query 文本(内部自动 fake_embed),
        也可以直接传 query_vec(省 CPU,做多次索引对比时复用同一个查询向量)。
        """
        if self.index is None:
            raise RuntimeError("还没建索引,请先调用 build_index('flat'/'ivf')")

        qv = query_vec if query_vec is not None else fake_embed(query)
        rows, scores = self.index.search(qv, k)   # 索引给出行号和相似度
        out = []
        # 根据行号,从 4 个平行列表里拼回完整信息
        for row, s in zip(rows, scores):
            out.append({
                "row":   row,
                "id":    self.row_ids[row],
                "score": s,
                "text":  self.texts[row],
                "meta":  self.metas[row],
            })
        return out

    # ─────────────────── Step 6:持久化(存磁盘) ───────────────────
    # 磁盘文件布局(存两个 JSON,方便你直接打开看内部长啥样):
    #   <dir>/db_meta.json  ← 数据层:dim / 模型名 / 4 个平行数组(含 vectors)
    #   <dir>/index.json    ← 索引层:Flat 的 vectors,或 IVF 的 centroids+buckets
    #
    # 真实产品(Chroma 为例):
    #   chroma_db/chroma.sqlite3     ← metadata + texts + id 映射
    #   chroma_db/chroma-embeddings/<uuid>/data_level_0.bin ← 向量数组
    #   chroma_db/chroma-embeddings/<uuid>/index.html      ← HNSW 图索引
    def save_local(self, directory: str) -> None:
        """把当前整个 DB(数据层 + 索引层) 序列化到磁盘目录 directory 下"""
        os.makedirs(directory, exist_ok=True)  # 目录不存在就建
        meta_path = os.path.join(directory, "db_meta.json")
        index_path = os.path.join(directory, "index.json")

        # ① 存数据层(4 个平行列表 + 头信息)
        db_meta = {
            "dim": self.dim,
            "embedding_model_name": self.embedding_model_name,
            "index_type": self.index_type,
            "n_rows": len(self.vectors),
            "row_ids": self.row_ids,
            "vectors": self.vectors,
            "texts": self.texts,
            "metas": self.metas,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(db_meta, f, ensure_ascii=False, indent=2)

        # ② 存索引层(调 Flat/IVF 各自的 state_dict 方法)
        with open(index_path, "w", encoding="utf-8") as f:
            state = self.index.state_dict() if self.index else {"type": "none"}
            json.dump(state, f, ensure_ascii=False, indent=2)

        print(f"💾 向量库已持久化到磁盘: {os.path.abspath(directory)}")
        print(f"   📄 {meta_path}  ({os.path.getsize(meta_path)//1024} KB)")
        print(f"   📄 {index_path} ({os.path.getsize(index_path)//1024} KB)")

    # ─────────────────── Step 6:从磁盘重新加载 ───────────────────
    @classmethod
    def load_local(cls, directory: str, expected_embedding_model: str | None = None) -> "ToyVectorDB":
        """
        从 save_local 存的目录,读回一个全新的 ToyVectorDB 对象。

        expected_embedding_model 是知识点里强调的「坑 1」:
          如果传了,就和 DB 里记录的 embedding_model_name 对比,
          不一致立即报错,避免「建库一个模型、查询另一个模型」把结果全搞乱。
        """
        meta_path = os.path.join(directory, "db_meta.json")
        index_path = os.path.join(directory, "index.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            db_meta = json.load(f)

        # ① 校验 embedding 模型名一致
        if expected_embedding_model and db_meta["embedding_model_name"] != expected_embedding_model:
            raise ValueError(
                f"Embedding 模型不匹配!库是 {db_meta['embedding_model_name']}, "
                f"你给的是 {expected_embedding_model},请删库重建。"
            )

        # ② 新建对象,把数据层字段全填回去
        db = cls(db_meta["dim"], db_meta["embedding_model_name"])
        db.row_ids = db_meta["row_ids"]
        db.vectors = db_meta["vectors"]
        db.texts = db_meta["texts"]
        db.metas = db_meta["metas"]
        db.index_type = db_meta["index_type"]

        # ③ 重建索引层
        global _GLOBAL_VECTORS
        with open(index_path, "r", encoding="utf-8") as f:
            idx_state = json.load(f)
        if idx_state["type"] == "flat":
            db.index = FlatIndex.from_state(idx_state)
        elif idx_state["type"] == "ivf":
            db.index = IVFIndex.from_state(idx_state)
        else:
            raise RuntimeError(f"load_local:未知索引类型 {idx_state['type']}")

        _GLOBAL_VECTORS = db.vectors
        print(f"📂 已从磁盘加载向量库:{os.path.abspath(directory)}")
        print(f"   共 {len(db.vectors)} 条,维度 {db.dim},索引 {db.index_type}")
        return db

    # ─────────────────── 调试工具:inspect() 打印内部快照 ───────────────────
    def inspect(self) -> None:
        """打印 DB 内部概览:条数/维度/索引类型/前6条样例/IVF桶大小分布"""
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
            # 打印 IVF 桶大小分布,看 K-Means 分得均匀不
            assert isinstance(self.index, IVFIndex)
            lens = sorted([len(b) for b in self.index.buckets])
            print(f"\n   🪣 IVF 桶大小:min={min(lens)}, median={lens[len(lens)//2]},"
                  f" max={max(lens)}, 桶数={len(self.index.buckets)}")
        print("=" * 70)


# ─────────────────────────────────────────────
# 2.5 最小完整例子:3 条 chunk 从头跑一遍(没数据量,不被 demo_full 刷屏)
# ─────────────────────────────────────────────
# 命令行跑:python 示例.py mini   (只跑这个最小例子)

MINI_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_vector_db_mini")

def demo_mini_walkthrough():
    """
    对应知识点创建 6 步,每一步都有打印,3 条数据一眼能看全。
    这是你第一次看这个文件的入口。
    """
    print("\n" + "#" * 70)
    print("# 🟢 最小例子:从零建一个「只有 3 条记录」的向量数据库")
    print("#" * 70)

    # Step 1:确定维度/模型
    print("\nStep 1 ✅ 确定模型:DIM=48, Embedding 模型名 = fake-embed-v1")
    db = ToyVectorDB(dim=DIM, embedding_model_name="fake-embed-v1")

    # Step 2:准备 3 条已经切好的 chunk(id, text, meta)
    chunks = [
        ("id_001", "员工请假需提前 3 天 OA 申请,部门主管审批生效",
                  {"source": "员工手册.pdf", "page": 8,  "tag": "请假"}),
        ("id_002", "报销需上传发票,月底前提交,财务 10 个工作日打款",
                  {"source": "员工手册.pdf", "page": 15, "tag": "报销"}),
        ("id_003", "出差一线城市住宿不超过 500 元一晚,高铁二等座",
                  {"source": "员工手册.pdf", "page": 22, "tag": "出差"}),
    ]
    print(f"\nStep 2 ✅ 准备好 {len(chunks)} 条 chunk:")
    for id_, text, meta in chunks:
        print(f"         {id_} / tag={meta['tag']} / {text[:28]}…")

    # Step 3:向量化 + 写入
    print("\nStep 3 ✅ 把 chunks 写进 DB(内部自动 fake_embed 转向量)")
    db.add_texts(chunks)
    db.inspect()  # 打印内部快照,看看 4 个平行数组长啥样

    # Step 4:ToyVectorDB 本身就是数据库
    # Step 5:选 Flat 建索引(3 条用 Flat 就行,100% 召回)
    print("\nStep 5 ✅ 建索引:选 Flat(暴力,因为只有 3 条没必要聚类)")
    db.build_index("flat")

    # Step 6a:存到磁盘
    print("\nStep 6a ✅ save_local:持久化到磁盘")
    db.save_local(MINI_SAVE_DIR)

    # Step 6b:模拟关进程 + 重新加载
    print("\nStep 6b ✅ del db → load_local:模拟「第二天重启程序,数据库还在」")
    del db
    db2 = ToyVectorDB.load_local(MINI_SAVE_DIR, expected_embedding_model="fake-embed-v1")

    # Step 6c:查询
    print("\nStep 6c ✅ 查 2 个问题,验证语义检索")
    for q in ["怎么请假?", "报销怎么弄?"]:
        print(f"\n💬 查询: {q}")
        for r in db2.search(q, k=2):
            print(f"   score={r['score']:.4f}  tag={r['meta']['tag']:<3}  "
                  f"id={r['id']}")
            print(f"         原文: {r['text']}")
    print("\n🎉 全流程 6 步完成:准备 → 写 → 建索引 → 存 → 载 → 查")


# ─────────────────────────────────────────────
# 3. 造一批「模拟 chunk 数据」(120 条,6 个主题混合)
# ─────────────────────────────────────────────

def build_mock_chunks(n_per_topic: int = 20) -> list[tuple[str, str, dict]]:
    """
    生成 mock 数据:
      6 个主题(请假/报销/出差/会议室/薪酬/打卡),每个主题 n_per_topic 条,
      用 {n} 插不同数字让每条不重复但语义高度相近,
      用于演示 IVF 的 K-Means 能自动按主题聚桶。
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
# 4. Demo 函数集合:不同规模的演示
# ─────────────────────────────────────────────
# 命令行参数选择运行哪个:
#   python 示例.py mini    → 只跑 demo_mini_walkthrough (3 条,推荐第一次看)
#   python 示例.py full    → 只跑 demo_full_workflow  (120 条 + IVF)
#   python 示例.py cmp     → 只跑 demo_compare_flat_vs_ivf (Flat vs IVF 召回率对比)
#   python 示例.py disk    → 只跑 show_disk_files (看持久化文件内容)
#   python 示例.py all     → 全部跑一遍(默认)

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
    """
    两种索引召回率对比:
      Flat(暴力)= 100% 召回(标准答案)
      IVF(K-Means聚类)= 近似召回,看能把 Flat 的结果覆盖到多少

    做的事情:150 条数据,拿 20 条随机文本(本身就在库里)做查询。
      指标 1:Top-1 命中—— IVF 返回的第 1 条是不是就是 Flat 返回的第 1 条(完全一致)?
      指标 2:Top-3-in-Top-5 召回—— Flat 前 3 条,是不是都在 IVF 返回的前 5 条里?

    想自己实验:把 build_index("ivf", nlist=6, nprobe=1) 的 nprobe 调大到 4,
    再跑一遍 cmp,看两个指标是不是显著提升。这就是 IVF 的「速度 ↔ 召回率」旋钮。
    """
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
    """
    把 save_local 写出去的两个 JSON 文件的前 500 字节打印出来预览。
    知识点 Layer 1(数据层 + 索引层) 的磁盘布局:
        db_meta.json → 数据层:头信息(dim/模型名) + row_ids / vectors / texts / metas
        index.json   → 索引层:Flat 的 vectors,或 IVF 的 centroids / buckets
    你可以直接用编辑器打开这两个文件,真的就是 JSON,能看懂。
    """
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
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    # 清空旧的持久化目录(避免之前 mini/full 互相污染)
    for d in [SAVE_DIR, MINI_SAVE_DIR]:
        try:
            import shutil
            if os.path.isdir(d):
                shutil.rmtree(d)
        except Exception:
            pass

    if arg in ("all", "mini"):
        demo_mini_walkthrough()
    if arg in ("all", "full"):
        demo_full_workflow()
    if arg in ("all", "cmp"):
        demo_compare_flat_vs_ivf()
    if arg in ("all", "disk"):
        # show_disk_files 依赖 demo_full_workflow 先跑(它写 my_vector_db)
        if not os.path.isdir(SAVE_DIR):
            demo_full_workflow()
        show_disk_files()
