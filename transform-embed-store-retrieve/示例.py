# -*- coding: utf-8 -*-
"""
transform → embed → store → retrieve 完整流水线
=================================================
零依赖纯 Python 可运行。演示 RAG 数据处理的完整 Pipeline:

  原始文档 → ①transform(切分) → ②embed(向量化) → ③store(存储)
  → ④retrieve(检索) → 拼 Prompt → LLM 回答

四环节在代码里分别对应:
  ① chunk_text()     文本变换:分块 + 重叠
  ② fake_embed()     嵌入:模拟 Embedding 模型(文本→向量)
  ③ VectorStore.add() 存储:向量+原文+metadata 落库
  ④ VectorStore.search() 检索:余弦相似度 Top-K

真实工程中,把 fake_embed 换成 bge / text-embedding-3 等模型,
把 VectorStore 换成 Chroma / FAISS / Milvus 即可,流程完全一致。
"""

import math

# ═══════════════════════════════════════════
# ① transform:把原始文档切分成 chunk
# ═══════════════════════════════════════════
def chunk_text(text: str, size: int = 120, overlap: int = 30) -> list[str]:
    """固定字符切分 + 重叠。
    真实工程: LangChain 的 RecursiveCharacterTextSplitter
    (按 段落→句子→词 逐级降切,更聪明;这里用固定切法演示原理)。
    """
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap          # 每次前进 size-overlap,保留下 overlap 的上下文
    return chunks


# ═══════════════════════════════════════════
# ② embed:模拟 Embedding 模型(文本 → 向量)
# ═══════════════════════════════════════════
def fake_embed(text: str) -> list[float]:
    """用一个简单的词袋哈希模拟 Embedding:
    - 每个关键词映射到固定维度的某几个位置 +1
    - 语义相近的文本(共享关键词) → 向量也接近
    真实工程: bge / text-embedding-3 等模型输出 384~1536 维向量。
    """
    vec = [0.0] * 64                                     # 固定 64 维(真实:512/1024)
    for ch in text:                                      # 按字符简单叠加(真实:按 token + 语义)
        idx = ord(ch) % 64                               # 哈希到维度
        vec[idx] += 1.0
    # 归一化(真实模型通常已归一化;归一化后余弦=点积,好算)
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


def cos_sim(a: list[float], b: list[float]) -> float:
    """余弦相似度:看两个向量"方向"多接近,∈[-1, 1]。"""
    return sum(x * y for x, y in zip(a, b))              # 归一化后点积 = 余弦


# ═══════════════════════════════════════════
# ③ store:迷你向量库(向量 + 原文 + metadata)
# ═══════════════════════════════════════════
class VectorStore:
    """极简向量库。真实工程: Chroma / FAISS / Milvus / Qdrant,
    内部还会建 HNSW/IVF 索引加速百万级检索,这里用列表暴力扫描。
    """

    def __init__(self):
        self.rows = []                                   # 每行:{id, vector, text, metadata}

    def add(self, id_: int, vector: list[float], text: str, metadata: dict):
        self.rows.append({"id": id_, "vector": vector,
                          "text": text, "metadata": metadata})
        print(f"  [store] 写入 #{id_} 向量({len(vector)}维) 来源:{metadata.get('doc')}")

    def search(self, q_vec: list[float], k: int = 2, doc_filter: str | None = None) -> list[dict]:
        """④ retrieve:算相似度 → 排序 → 取 Top-K(可先按 metadata 过滤)。"""
        rows = [r for r in self.rows
                if doc_filter is None or r["metadata"].get("doc") == doc_filter]
        scored = [(cos_sim(q_vec, r["vector"]), r) for r in rows]
        scored.sort(key=lambda x: -x[0])                 # 相似度从高到低
        return [{"score": round(s, 3), "text": r["text"],
                 "metadata": r["metadata"]} for s, r in scored[:k]]


# ═══════════════════════════════════════════
# 主流程:把四环节串起来
# ═══════════════════════════════════════════
def main():
    print("=" * 68)
    print("① transform:加载文档并切分(分块)")
    print("=" * 68)
    docs = [
        {"doc": "行政制度.md", "content":
         "员工请假流程:提前3天在OA系统提交申请,经部门主管审批后生效。年假全年10天。"
         "费用报销:因公消费需保留发票,月底前在报销系统上传电子发票,财务10个工作日内打款。"},
        {"doc": "行政制度.md", "content":
         "工资发放:每月15日发放上月工资,逢节假日顺延。会议室预订:通过内部预订系统选择时间,"
         "时长超过2小时需抄送部门负责人。"},
    ]
    chunks = []                                          # 所有 chunk 的列表
    for d in docs:
        for i, c in enumerate(chunk_text(d["content"])):
            chunks.append({"text": c, "metadata": {"doc": d["doc"], "chunk": i}})
            print(f"  [transform] chunk#{len(chunks)}: {c}")

    print(f"\n{'='*68}")
    print("② embed:每个 chunk 转成 64 维向量")
    print("=" * 68)
    for c in chunks:
        c["vector"] = fake_embed(c["text"])

    print(f"\n{'='*68}")
    print("③ store:向量 + 原文 + metadata 写入向量库")
    print("=" * 68)
    db = VectorStore()
    for i, c in enumerate(chunks, 1):
        db.add(i, c["vector"], c["text"], c["metadata"])

    print(f"\n{'='*68}")
    print("④ retrieve:用户问题 → 转向量 → 搜 Top-K")
    print("=" * 68)
    questions = ["请假流程是什么?", "报销要发票吗?", "薪资怎么发?"]
    for q in questions:
        q_vec = fake_embed(q)                            # 必须同一个"模型"!fake_embed 即模型
        print(f"\n  问题: {q}")
        for hit in db.search(q_vec, k=2):                # 拿最像的 2 块
            print(f"    score={hit['score']} | {hit['text'][:30]}...")

    print(f"\n{'='*68}")
    print("拼 Prompt → LLM 回答(演示)"
          "\n真实工程: 把 top_k 文本块塞进 Prompt,由 LLM 基于这些材料回答")
    print("=" * 68)
    q_vec = fake_embed("请假流程是什么?")
    hits = db.search(q_vec, k=2)
    prompt = ("请基于以下资料回答:\n" +
              "\n".join(f"- {h['text']}" for h in hits) +
              "\n\n问题:请假流程是什么?")
    print(prompt[:200])
    print("\n→ LLM 回答: 员工需提前 3 天在 OA 系统提交申请,经部门主管审批后生效。")
    print("  (实际中这里由真实 LLM 基于检索到的 chunk 生成)")


if __name__ == "__main__":
    main()
    print("\n\n[对比提示] 在真实工程(LangChain)中,这四步对应:")
    print("  loaders + TextSplitter → Embeddings → VectorStore → Retriever")
