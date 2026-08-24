"""
RAG 真实例子 —— 公司知识库问答机器人
=====================================================

【真实业务场景】
某公司把《员工手册》《报销制度》《出差规定》《入职指南》等内部文档
喂给 RAG 系统,做了一台「HR 问答机器人」,挂在钉钉/飞书群里:

    员工提问:「出差住宿一晚能报多少钱?」
    机器人回答:「根据《出差规定.pdf》规定:一线城市不超过 500 元/晚……」
    (答案来自公司文档,而不是大模型瞎编的)

【为什么不能直接问 LLM?】
    1. LLM 没读过你们公司的文档 —— 问「报销几天打款」它只能编
    2. RAG = 先从知识库检索最相关的 3 段 → 再让 LLM 基于材料回答
       → 答案有出处、能防幻觉、文档更新 = 答案跟着更新(不用重训模型)

【RAG 完整闭环(本文件从头到尾跑一遍)】
    建库:  文档 → 切分 chunk → 向量化 → 存进向量库
    问答:  问题 → 向量化 → 检索 Top-K → 拼 prompt → LLM 生成答案

【和 示例.py 的关系】
    示例.py   = 内部零件(手搓向量数据库,讲「检索」这一步)
    真实例子.py = 完整产品(文档 → 检索 → 喂给 LLM → 出答案)

【运行方式】零依赖,直接:  python 真实例子.py
"""

import math
import random
import sys
from typing import Any

# Windows 控制台默认 GBK,强制用 UTF-8 输出(否则 emoji / 生僻字会报错)
# 用 getattr 调用 reconfigure(部分 Python 环境该属性不在静态类型里,避开类型误报)
try:
    getattr(sys.stdout, "reconfigure")(encoding="utf-8")
except Exception:
    pass

# ═════════════════════════════════════════════════════════
# 第 0 步:模拟 Embedding 模型(把文本变成一串数字)
# ═════════════════════════════════════════════════════════
# 真实场景:sentence-transformers / OpenAI text-embedding-3 等
# 这里用「字 + 相邻双字 哈希累加 + 归一化」模拟:
#   · 每个字/双字组合映射成一个固定向量
#   · 文本向量 = 里面所有字/双字向量的累加
#   · 相同词语越多 → 向量越像 → 相似度越高
# 「双字」特征很关键:比如「差旅」「报销」「住宿」会被当成整体特征,
# 这让检索效果比纯单字哈希(之前示例.py 用的)好很多,更接近真 embedding。

VEC_DIM = 64
_CHAR_VEC_CACHE: dict[str, list[float]] = {}


def _char_vec(feat: str) -> list[float]:
    """一个特征(单字或双字)→ 一个固定向量(用特征自己做随机种子,保证结果稳定)"""
    if feat in _CHAR_VEC_CACHE:          # 例: 之前算过 '报' → 命中缓存
        return _CHAR_VEC_CACHE[feat]     # 直接返回缓存里的向量(跳过下面所有重算)
    # 用 FNV-1a 算法自算稳定哈希(不能用 Python 内置 hash():
    # 它对字符串每次运行随机变化,会导致每次运行检索结果不一样)
    h = 2166136261                       # 初始哈希值(固定常数,FNV 偏移基数)
    for b in feat.encode("utf-8"):       # 例: '报' 编码成 2 个字节,逐个参与哈希
        h ^= b  # 等价于: h = (h ^ b) & 0xFFFFFFFF   # 字节值与哈希异或
        h = (h * 16777619) & 0xFFFFFFFF               # 乘质数并截断到 32 位
    rng = random.Random(h)               # 用哈希做随机种子 → 同一特征每次随机序列完全一致
    vec = [rng.uniform(-1.0, 1.0) for _ in range(VEC_DIM)]  # 例: '报' → 64 个 [-1,1] 的随机数,前 6 个 ≈ [-0.2, 0.461, -0.585, 0.152, -0.897, 0.586]
    _CHAR_VEC_CACHE[feat] = vec          # 结果存进缓存,下次直接命中
    return vec


def _text_features(text: str) -> list[str]:
    """把文本拆成特征列表:所有单字 + 所有相邻双字"""
    features = [ch for ch in text if ch.strip()]     # ① 单字: "差旅报销" → ['差','旅','报','销']
    features += [text[i:i+2] for i in range(len(text) - 1) if text[i:i+2].strip()]  # ② 相邻双字: → ['差旅','旅报','报销']
    return features        # → 最终 = ['差','旅','报','销','差旅','旅报','报销'](7 个特征)


def embed(text: str) -> list[float]:
    """text → 向量。真实写法:sentence_transformers.encode(text)"""
    vec = [0.0] * VEC_DIM                     # ① 64 个 0 的全零向量,准备累加
    for feat in _text_features(text): # ['差','旅','报','销','差旅','旅报','报销'] = 7 个特征
        fv = _char_vec(feat)                  # 每个特征取自己的 64 维向量
        for i in range(VEC_DIM):
            vec[i] += fv[i]                   # ② 64 维逐位累加 → vec = 7 个特征向量之和
    norm = math.sqrt(sum(v * v for v in vec)) or 1e-9  # ③ 模长(空文本=0 时用 1e-9 兜底,防除零)
    return [v / norm for v in vec]            # ④ 每个分量除以模长 → 归一化(方向不变,长度=1)


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度(向量已归一化,直接点积即可)"""
    # 两个归一化向量做点积:对应分量相乘再求和(向量已归一化,|a|·|b|=1,所以 cos=a·b)
    return sum(x * y for x, y in zip(a, b))


# ═════════════════════════════════════════════════════════
# 第 1 步:加载文档 + 切分(Chunking)
# ═════════════════════════════════════════════════════════
# 真实场景:PyPDF / Unstructured / 爬虫 读文档,
#          RecursiveCharacterTextSplitter 按 300~500 字切块,带重叠和元信息
# 这里简化:每个文档直接给了切好的小块

def load_and_split() -> list[dict[str, Any]]:
    """模拟「读取公司文档 → 切分」,返回 [{text, meta}, ...]"""
    documents = [
        {
            "source": "员工手册.pdf",
            "chunks": [
                "员工请假流程:提前 3 天在 OA 系统提交申请,经部门主管审批后生效。年假全年 10 天。",
                "工资发放:每月 15 日发放上月工资,逢节假日顺延至下一工作日,工资条在 HR 系统查看。",
                "试用期制度:新员工试用期 3 个月,表现优秀可提前转正。",
                "加班政策:工作日加班按 1.5 倍工资计算,需提前在 OA 报备并打卡留痕。",
            ],
        },
        {
            "source": "报销制度.pdf",
            "chunks": [
                "费用报销流程:因公消费需保留发票,当月月底前上传电子发票,财务 10 个工作日内打款。",
                # 真实项目里用户说「打车费」,真 embedding 模型能语义匹配上「交通费」;
                # 这里为了演示稳定,材料里直接写了「打车费」
                "差旅报销:出差产生的打车费、交通费、住宿费、餐费均可报销,需附行程单和发票。",
                "通讯补贴:每月话费补贴 200 元,凭发票实报实销。",
            ],
        },
        {
            "source": "出差规定.pdf",
            "chunks": [
                "出差住宿标准:一线城市住宿一晚不超过 500 元,其他城市不超过 350 元,高铁选二等座。",
                "出差申请:出差前需在 OA 提交出差申请单,注明事由、日期和预算。",
                "境外出差:需提前 2 周申请,经总经理审批,费用标准另行核定。",
            ],
        },
        {
            "source": "入职指南.docx",
            "chunks": [
                "入职体检:新员工需在报到后一周内到指定三甲医院完成体检,费用由公司承担。",
                "入职资料:报到当天需携带身份证、毕业证、离职证明、一寸照片两张。",
            ],
        },
        {
            "source": "公积金说明.pdf",
            "chunks": [
                "公积金缴纳:按上一年度月平均工资的 12% 缴纳,个人与公司各承担一半。",
                "公积金提取:租房可每季度提取一次,在市民中心或公积金 App 申请。",
            ],
        },
    ]
    result = []                                  # 结果列表,初始为空
    for doc in documents:                        # 遍历 5 份文档(员工手册/报销制度/出差规定/入职指南/公积金说明)
        for i, text in enumerate(doc["chunks"]): # i = 该文档内的 chunk 序号(0,1,2…)
            result.append({
                "text": text,                    # 正文
                "meta": {"source": doc["source"], "chunk": i},  # 来源文档 + 序号
            })
    return result   # → 14 个 chunk:员工手册4 + 报销制度3 + 出差规定3 + 入职指南2 + 公积金说明2


# ═════════════════════════════════════════════════════════
# 第 2 步:模拟 LLM(真实场景 = 调 OpenAI / DeepSeek / 本地模型)
# ═════════════════════════════════════════════════════════
class MockLLM:
    """
    模拟大模型:给它一段「资料 + 问题」的 prompt,它输出答案。
    真实场景是调 chat 接口,这里规则化模拟 LLM 的「阅读材料 → 判断能否回答」:
      · 检索到的材料里出现了问题的关键词 → 摘取对应材料回答(带出处)
      · 材料里找不到相关内容            → 承认不知道(这是 RAG 防幻觉的关键!)
    """

    # 问题里的这些词不算「关键词」(真实场景是 LLM 的语义理解,这里用停用词模拟)
    _STOPWORDS = {"的", "了", "吗", "能", "有", "都", "什么", "怎么", "多少",
                  "一", "个", "我", "他", "请", "问", "下", "呀", "呢"}

    def answer(self, prompt: str, question: str, hits: list[dict[str, Any]]) -> str:
        if not hits:                              # 检索结果为空(知识库没找到任何相关)
            return "知识库中未找到相关内容,请咨询 HR 或查阅公司文档。"  # → 直接认不知道,不硬编

        # 提取问题里的「关键词」(双字词组,去掉停用词)
        q_keywords = [f for f in _text_features(question)
                      if len(f) >= 2 and f not in self._STOPWORDS]
        # 例: "出差住宿一晚能报多少钱?" 拆出 23 个特征 → 滤掉停用词(能/多少/一)和单字
        #   → ['出差','差住','住宿','宿一','一晚','晚能','能报','报多','少钱','钱?'] 共 10 个

        # 模拟 LLM 逐条「阅读」检索到的材料:找到覆盖了关键词最多的那条
        best, best_hit = 0, None                  # best = 最高重合数, best_hit = 对应材料
        for h in hits:                            # 逐条看检索到的材料
            hit_words = set(_text_features(h["text"]))       # 材料也拆成特征集合
            overlap = sum(1 for w in q_keywords if w in hit_words)  # 与问题重合几个关键词
            if overlap > best:                    # 比之前的都多 → 暂时胜出
                best, best_hit = overlap, h
        # 例(第一个问题): Top-1 材料「出差住宿标准…」 overlap = 5 → best = 5

        # 材料与问题有 ≥2 个共同关键词 → 相关,可以回答
        if best >= 2 and best_hit is not None:
            return f"根据《{best_hit['meta']['source']}》规定:{best_hit['text']}"  # → 带出处的答案
        return "知识库中未找到相关内容,请咨询 HR 或查阅公司文档。"  # overlap < 2 → 承认不知道


# ═════════════════════════════════════════════════════════
# 第 3 步:ToyRAG —— 完整 RAG 引擎(建库 + 问答闭环)
# ═════════════════════════════════════════════════════════
class ToyRAG:
    def __init__(self, llm: MockLLM):
        self.llm = llm
        self.chunks: list[dict[str, Any]] = []      # 原文 + 元信息
        self.vectors: list[list[float]] = []        # 每条 chunk 对应的向量
        self.idf: dict[str, float] = {}         # 特征 → 权重(建库时统计)

    # ---------- 带权重的向量化(模拟 TF-IDF,提高关键词区分度) ----------
    # 真实 embedding 模型内部也做了类似的事:重要的低频词(报销/差旅/公积金)
    # 贡献更大,高频虚词(的/了/是)贡献更小。这里手动实现。
    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * VEC_DIM                        # 全零向量,准备加权累加
        for feat in _text_features(text):
            # 见过该特征 → 用它的 IDF 权重;没见过(通常是查询里的生词)→ 给中高权重
            w = self.idf.get(feat, 1.5)              # 例: '报销' → 2.5404(权重高);'钱'(生词)→ 1.5
            fv = _char_vec(feat)                     # 特征自己的 64 维向量
            for i in range(VEC_DIM):
                vec[i] += fv[i] * w                  # 加权累加:关键词向量被放大,虚词被缩小
        norm = math.sqrt(sum(v * v for v in vec)) or 1e-9
        return [v / norm for v in vec]               # 归一化(与 embed() 相同,只是多了权重)

    # ---------- 建库(离线做一次) ----------
    def ingest(self, docs: list[dict[str, Any]]) -> None:
        from collections import Counter              # 计数器:不存在的 key 自动从 0 开始
        n = len(docs)                                # → n = 14(本库共 14 个 chunk)
        # ① 统计每个特征出现在多少个 chunk 里(文档频率)
        df = Counter()                               # 空计数器: {'词': 出现在几个 chunk}
        for d in docs:                               # 遍历每个 chunk
            for feat in set(_text_features(d["text"])):  # set 去重:同一 chunk 里重复出现只算 1 次
                df[feat] += 1                        # 例: df['报销']=2(2 个 chunk 有), df['晚']=1
        # ② 算 IDF:出现越少 → 权重越高
        self.idf = {f: math.log(n / (1 + c)) + 1.0 for f, c in df.items()}
        # 例(n=14): c=1 → log(14/2)+1 ≈ 2.95(只 1 个 chunk 有 → 权重最高)
        #           c=2 → log(14/3)+1 ≈ 2.54;  c=14 → log(14/15)+1 ≈ 0.93(每个 chunk 都有 → 最低)
        # ③ 向量化 + 存储
        for d in docs:
            self.chunks.append(d)                    # 存原文(检索命中后展示给 LLM 用)
            self.vectors.append(self._embed(d["text"]))  # 存 64 维加权向量(和 chunks 下标一一对应)
        print(f"📥 建库完成:共 {len(self.chunks)} 个 chunk,"
              f"来源 {len({c['meta']['source'] for c in self.chunks})} 份文档")
        # 实际输出 → 📥 建库完成:共 14 个 chunk,来源 5 份文档

    # ---------- 检索(向量相似度 Top-K) ----------
    def retrieve(self, question: str, k: int = 3) -> list[dict[str, Any]]:
        qv = self._embed(question)                   # 问题也向量化(用建库时同一套 IDF 权重)
        scored: list[dict[str, Any]] = []            # 打分结果列表
        for i, vec in enumerate(self.vectors):       # 遍历库里每个 chunk 的向量(共 14 个)
            scored.append({
                "score": round(cosine(qv, vec), 4),  # 问题向量和 chunk 向量的余弦相似度
                "text": self.chunks[i]["text"],      # 例: "出差住宿…" vs 出差规定 → 0.289
                "meta": self.chunks[i]["meta"],
            })
        scored.sort(key=lambda x: x["score"], reverse=True)  # 相似度从高到低排序
        return scored[:k]                            # 取前 k 个(默认 3)返回

    # ---------- 问答完整闭环(4 步) ----------
    def ask(self, question: str, k: int = 3) -> str:
        print(f"\n💬 用户提问:{question}")
        print("-" * 64)

        # ① 检索:问题向量化 → 找最像的 K 个 chunk
        hits = self.retrieve(question, k)
        print("① 检索 Top-%d(向量相似度):" % k)
        for h in hits:
            print(f"   score={h['score']:.4f}  [{h['meta']['source']}] {h['text'][:24]}…")

        # ② 组装 prompt:把检索到的材料塞进 prompt(这就是 RAG 的"增强"!)
        context = "\n\n".join(
            f"[资料{i+1}](来源:{h['meta']['source']})\n{h['text']}"
            for i, h in enumerate(hits)
        )
        prompt = (
            "你是一名 HR 问答助手。请只根据下面提供的公司内部资料回答问题,不要编造。\n"
            "如果资料里找不到答案,请直接说「知识库中未找到相关内容,请咨询 HR」。\n\n"
            f"{context}\n\n"
            f"问题:{question}"
        )
        print("\n② 组装 prompt(发给 LLM):")
        for line in prompt.splitlines():
            print(f"   | {line}")

        # ③ LLM 基于材料生成答案
        answer = self.llm.answer(prompt, question, hits)
        print("\n③ LLM 生成答案:", answer)
        return answer


# ═════════════════════════════════════════════════════════
# 主流程:建库 → 连续问几个真实问题
# ═════════════════════════════════════════════════════════
def main():
    print("🏢 场景:公司 HR 知识库问答机器人(员工手册 + 报销 + 出差 + 入职 + 公积金)")
    print("=" * 64)

    # 建库:文档 → 切分 → 向量化(一次,之后新文档进来增量 ingest)
    docs = load_and_split()                 # → 14 个 chunk
    rag = ToyRAG(llm=MockLLM())             # 建引擎(LLM 用规则模拟)
    rag.ingest(docs)                        # → 📥 建库完成:共 14 个 chunk,来源 5 份文档

    # 连续问答:覆盖「语义召回」「同义词」「无答案」三种情况
    rag.ask("出差住宿一晚能报多少钱?")     # → 命中 出差规定(500 元/晚),Top-1 score=0.289
    rag.ask("打车费能报销吗?")             # → 语义命中 差旅报销(「打车费」≈「交通费」),Top-1 score=0.3448
    rag.ask("公积金怎么提取?")             # → 命中 公积金说明,Top-1 score=0.5417
    rag.ask("年会抽奖的奖品都有什么?")     # → 知识库里没有 → 承认不知道(防幻觉)


if __name__ == "__main__":
    main()

    # ═════════════════════════════════════════════════════
    # 真实生产代码对照(装了库再跑:pip install langchain-chroma langchain-huggingface)
    # ═════════════════════════════════════════════════════
    print("\n" + "=" * 64)
    print("📦 真实生产版本对照(LangChain + Chroma + 大模型 API)")
    print("=" * 64)
    try:
        # ---- 新版 LangChain 写法 ----
        from langchain_chroma import Chroma  # type: ignore[reportMissingImports]
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore[reportMissingImports]
        from langchain_core.prompts import ChatPromptTemplate  # type: ignore[import-not-found]
        from langchain_core.output_parsers import StrOutputParser  # type: ignore[import-not-found]
    except ImportError:
        print("[跳过] 未安装 langchain-chroma / langchain-huggingface,装上后可运行")
    else:
        # 建库:embedding 用真模型,存到本地磁盘
        docs = load_and_split()  # main() 里的 docs 在这个作用域取不到,重新加载
        emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
        texts = [d["text"] for d in docs]
        metas = [d["meta"] for d in docs]
        db = Chroma.from_texts(
            texts=texts, embedding=emb, metadatas=metas,
            persist_directory="./chroma_db",
        )

        # 问答:检索 → 拼 prompt → 大模型回答
        # (真实环境填好 key 后,把下面这行换成 ChatOpenAI(model="gpt-4o") / DeepSeek 等)
        # llm = ChatOpenAI(model="gpt-4o-mini", api_key="sk-xxx")
        # hits = db.similarity_search("出差住宿一晚能报多少钱?", k=3)
        # context = "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in hits)
        # prompt = ChatPromptTemplate.from_messages([
        #     ("system", "只根据资料回答,不知道就说不知道"),
        #     ("human", "{context}\n\n问题:{question}"),
        # ])
        # answer = (prompt | llm | StrOutputParser()).invoke(
        #     {"context": context, "question": "出差住宿一晚能报多少钱?"})
        # print(answer)

        # 这里只演示检索(不调付费 API):
        hits = db.similarity_search("出差住宿一晚能报多少钱?", k=2)
        for d in hits:
            print(f"   [真实检索命中] {d.metadata['source']}: {d.page_content}")
