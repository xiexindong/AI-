# 提高 RAG 准确率的 10 个策略:从 Simple RAG 到 Contextual Compression

> 来源：RAG 进阶策略路线图。
> 面试场景：**"除了基础 RAG,你还知道哪些提升准确率的方法?"**

---

## 一、10 个策略一览表

| # | 策略 | 核心动作 | 解决什么问题 |
|---|---|---|---|
| 01 | **Simple RAG** | 切分 → embedding → 检索 → LLM | baseline |
| 02 | **Semantic Chunking** | 按语义边界切分 | 解决硬切导致的语义断裂 |
| 03 | **Small-to-Big Retrieval** | 小 chunk 检索,大 chunk 拼 prompt | 解决小 chunk 缺上下文、大 chunk 不精准 |
| 04 | **Context Enriched Retrieval** | 检索时引入相邻 chunk / 摘要 | 解决单 chunk 信息孤立 |
| 05 | **Contextual Chunk Headers** | 给 chunk 加标题/来源头信息 | 让 chunk 自包含上下文 |
| 06 | **Document Augmentation** | 生成摘要/问答对/关键词一起入库 | 增强召回触角 |
| 07 | **Query Transformation** | 改写/扩展/分解 query | 解决用户 query 和文档表达不一致 |
| 08 | **Reranker** | 精排 Top-K | 解决向量排序不够准 |
| 09 | **RSE(Relevant Segment Extraction)** | 从大 chunk 里抽出最相关片段 | 减少无关信息干扰 LLM |
| 10 | **Contextual Compression** | 压缩检索结果 | 减少 context 噪音、省 token |

> **一句话路线图:先保证 chunk 切得好 → 再让 chunk 有上下文 → 再让 query 更好匹配 → 最后排序和压缩做精排。**

---

## 二、逐个拆解

### 01 Simple RAG

```
文档 → 切分 → embedding → 向量库 → 检索 Top-K → 拼 prompt → LLM
```

- **作用**:baseline,跑通 RAG 闭环
- **准确率瓶颈**:切分硬、query 不准、Top-K 排序质量一般
- **面试定位**:"基础 RAG 只能做到 60 分,后面的 9 个策略都是往上补。"

---

### 02 Semantic Chunking(语义切分)⭐

**核心**:不按固定字数切,而是**用 embedding 看相邻句子语义是否突变**,在突变处下刀。

```python
# 简化思路:相邻句子向量点积骤降 → 说明话题变了 → 在这里切
sentence_vec[i] vs sentence_vec[i+1]
  cosine 相似度 < 阈值 → 切一刀
```

- **为什么提高准确率**? chunk 内部语义内聚,不会把"请假流程"和"工资发放"硬拼在一起
- **代价**:入库前要多跑 embedding,速度慢
- **适合**:长文档、段落不明显、自然语言文本

> 面试话术:"语义切分把 chunk 从'等长片段'升级成'语义单元',向量空间的聚类性更好,召回自然更准。"

---

### 03 Small-to-Big Retrieval(小查大用)⭐

**核心**:

```
检索阶段:用小 chunk(语义集中,匹配精准)
生成阶段:把命中 chunk 所在的大 chunk / 父块喂给 LLM(上下文完整)
```

```python
# 索引库
子块:"费用报销流程:所有因公消费需保留发票"  → 父块:整段报销流程

用户问:「报销需要什么材料?」
  子块命中 → 召回父块 → LLM 看到完整上下文
```

- **为什么提高准确率**? 小 chunk 命中率高,大 chunk 给 LLM 的上下文完整,鱼和熊掌兼得
- **面试定位**:这是《切分策略.md》里"父子分块"的工程落地

---

### 04 Context Enriched Retrieval(上下文增强检索)

**核心**:检索时,不只看单个 chunk,而是把**相邻 chunk / 同段落 / 文档摘要**一起编码或一起打分。

```python
# 做法 A:检索时把命中 chunk 的前一段、后一段一起拼成"上下文块"喂给 LLM
# 做法 B:把 chunk 和它的 2-hop 邻居一起编码,增强语义表达
```

- **为什么提高准确率**? 单 chunk 可能缺主语、缺背景,邻块补全后语义更完整
- **和 overlap 的区别**:overlap 是切分阶段重复,这里是检索阶段动态补上下文

---

### 05 Contextual Chunk Headers(上下文头信息)

**核心**:给每个 chunk 加一段"我是谁、从哪来"的头部,让 chunk 自包含上下文。

```
原始 chunk:
  "每月 15 日发放上月工资,逢节假日顺延。"

加 header:
  "来源:员工手册.pdf / 章节:薪酬制度 / 原文:每月 15 日发放上月工资,逢节假日顺延。"
```

- **为什么提高准确率**? LLM 看到 header 就知道"这是薪酬制度",不会和其他"15 日"混淆;检索时 header 也参与 embedding,语义更丰富
- **注意**:header 不能太长,否则喧宾夺主;通常用"文档名 + 章节路径"

---

### 06 Document Augmentation(文档增强)

**核心**:不只入库原文,还把**摘要、问答对、关键词、同义词、反向链接**一起向量化存进去。

```python
# 原始 chunk
"年假全年 10 天,试用期员工按比例折算。"

# 同时入库的增强内容
摘要: "年假天数的计算规则"
问答对: "年假多少天? → 全年 10 天"
关键词: ["年假", "带薪休假", "假期"]
```

- **为什么提高准确率**? 用户 query 的表达方式多样,增强内容扩大召回触角
- **代价**:库变大,更新时维护成本高
- **适合**:FAQ、客服、术语多变的领域

---

### 07 Query Transformation(查询改写)⭐

**核心**:用户 query 不一定能直接匹配文档,先改写/扩展/分解 query,再检索。

常见做法:

| 方法 | 做法 | 例子 |
|---|---|---|
| **Query Expansion** | 扩展同义词 | "打车费" → "打车费 交通费 差旅费" |
| **HyDE** | 让 LLM 先写个假设答案,用这个答案做检索 | 问题 → 生成答案 → embedding → 检索 |
| **Multi-Query** | 把一个问题拆成多个角度 | "报销流程" → "怎么报销"+"报销材料"+"报销多久" |
| **Sub-Query** | 复杂问题拆子问题 | "销售和财务的报销流程区别" → 分别查销售/财务 |

- **为什么提高准确率**? 把用户口语化/歧义化的问题,转成文档里更可能出现的表达
- **代价**:每次查询要调 LLM,延迟增加
- **适合**:用户问题短、歧义多、和文档用词差异大

---

### 08 Reranker(重排序)⭐

**核心**:向量召回 Top-K 后,用更强的交叉编码器(cross-encoder)再精排一次。

```
用户 query ──┐
             ├── Cross-Encoder ──→ 相关度分数 ──→ 重排 ──→ 取 Top-3
候选 chunk ──┘
```

- **为什么提高准确率**? 双编码器(embedding)只能算"语义相似",交叉编码器能算"query 和候选的精确匹配",更准确
- **代价**:延迟 + 计算成本
- **做法**:向量召回 Top-100 → rerank 取 Top-3

> 详见《RAG落地的4个关键问题.md》"问题 4"。

---

### 09 RSE(Relevant Segment Extraction,相关片段提取)

**核心**:召回的 chunk 可能仍然包含无关内容,用模型从 chunk 中**只抽和问题相关的句子/片段**,再喂给 LLM。

```
召回 chunk:
  "员工请假流程:提前 3 天在 OA 系统提交申请,经部门主管审批后生效。年假全年 10 天。"

用户问:「年假多少天?」
  RSE 抽取: "年假全年 10 天。"   ← 只把最相关片段给 LLM
```

- **为什么提高准确率**? 减少无关信息对 LLM 的干扰,降低幻觉
- **和 Contextual Compression 的区别**:RSE 是"抽取",Compression 是"压缩/改写"

---

### 10 Contextual Compression(上下文压缩)

**核心**:把检索到的长 context **压缩成更短、但保留关键信息的摘要**,再喂给 LLM。

```
原始 context(3 个 chunk,共 1500 字) → 压缩模型 → 300 字精华摘要 → LLM
```

- **为什么提高准确率**? LLM context 有限,去掉噪音后注意力更集中;同时省 token
- **代价**:多一步压缩模型,可能丢失细节
- **适合**:长文档、Top-K 合并后 context 很长

---

## 三、组合使用路线图(从 baseline 到生产)

```
Step 1: Simple RAG 跑通
Step 2: 切分优化 + Contextual Chunk Headers
        (Semantic Chunking + Header)
Step 3: 上下文增强
        (Small-to-Big / Context Enriched Retrieval)
Step 4: 检索增强
        (Query Transformation + Document Augmentation)
Step 5: 精排与压缩
        (Reranker + RSE + Contextual Compression)
```

> **不是每个项目都要上满 10 个**。根据数据量、延迟预算、准确率要求,从后往前加。

---

## 四、面试常见追问

### Q1:Small-to-Big 和 Context Enriched Retrieval 有什么区别?

- **Small-to-Big**:检索和生成用不同粒度的 chunk(小 chunk 检索,大 chunk 生成),粒度不同
- **Context Enriched Retrieval**:检索时把命中 chunk 的邻居/摘要一起考虑,补充的是**上下文范围**

### Q2:Query Transformation 和 Document Augmentation 都为了解决"query 和文档对不上",有什么区别?

- **Query Transformation**:改 query,让 query 更像文档
- **Document Augmentation**:改文档/增强文档,让文档更容易被 query 命中
- **互补**:一个左对齐(query),一个右对齐(document)

### Q3:上了 Reranker 还要 RSE 吗?

- 职责不同。**Reranker 解决排序问题**,**RSE 解决 chunk 内部噪音问题**
- 组合:向量召回 Top-100 → Reranker 排 Top-10 → RSE 从 Top-10 里抽相关片段 → 喂 LLM

---

## 五、面试答题模板(30 秒版)

> "提高 RAG 准确率,我从 4 个层面优化:**切分层**用 Semantic Chunking + Contextual Chunk Headers,保证 chunk 语义完整、自包含上下文;**检索层**用 Small-to-Big 和 Context Enriched Retrieval 补全上下文,用 Query Transformation 和 Document Augmentation 解决 query 与文档表达不一致;**排序层**加 Reranker 精排 Top-K;**生成层**用 RSE 抽相关片段、Contextual Compression 压缩噪音 context。每一层改进都用 golden set 测 Recall@K 和答案质量,组合后 top-1 命中率能从 60% 提到 90% 以上。"

---

## 六、一句话总结

> "RAG 准确率不是单点优化出来的,而是**切分→检索→排序→生成**层层叠加。把这 10 个策略按'先解决 chunk 质量,再解决语义匹配,最后解决排序和噪音'的顺序组合使用,比盲目加 Reranker 有效得多。"
