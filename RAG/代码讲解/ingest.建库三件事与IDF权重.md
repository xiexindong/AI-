# `ingest` 函数解析:建库——统计词频、算 IDF、存向量

> 出处:`真实例子.py` 第 219~234 行。

```python
def ingest(self, docs: list[dict[str, Any]]) -> None:
    from collections import Counter
    n = len(docs)
    # ① 统计每个特征出现在多少个 chunk 里(文档频率)
    df = Counter()
    for d in docs:
        for feat in set(_text_features(d["text"])):
            df[feat] += 1
    # ② 算 IDF:出现越少 → 权重越高
    self.idf = {f: math.log(n / (1 + c)) + 1.0 for f, c in df.items()}
    # ③ 向量化 + 存储
    for d in docs:
        self.chunks.append(d)
        self.vectors.append(self._embed(d["text"]))
    print(f"📥 建库完成:共 {len(self.chunks)} 个 chunk,"
          f"来源 {len({c['meta']['source'] for c in self.chunks})} 份文档")
```

---

## 一、函数目的(一句话)

> **把一批 chunk 变成"可检索的索引":统计每个词的重要程度(IDF),再把每个 chunk 的原文 + 加权向量存进库。** 建库只在启动时做一次(离线),之后用户提问全靠这份索引检索。

它在整个 RAG 闭环中的位置:

```
离线:ingest(本文)→ 建好 chunks + vectors + idf
在线:retrieve(用户问题)→ 向量相似度 → 取 Top-3 → 交给 MockLLM 回答
```

---

## 二、逐行拆解

### 第 219 行:`def ingest(self, docs)`

- `docs`:切好的 chunk 列表,每个元素是 `{"text": 正文, "meta": {"source": 来源, "chunk": 序号}}`
- 无返回值,它把内容"装进" `self` 的三个属性:
  - `self.chunks`(原文)
  - `self.vectors`(向量)
  - `self.idf`(词权重表)

### 第 220 行:`from collections import Counter`

把计数器类引入局部作用域。

```python
c = Counter()
c['报销'] += 1   # 不存在的 key 自动从 0 开始
# Counter({'报销': 1})  —— 这就是它比普通 dict 方便的地方
```

### 第 221 行:`n = len(docs)`

记录 chunk 总数,第 228 行算 IDF 的分母要用。

### 第 223 行:`df = Counter()`

`df` = **document frequency(文档频率)**,记录"每个特征出现在多少个 chunk 里"。

### 第 224~226 行:统计 df(这段是重点!)

```python
for d in docs:                          # 遍历每个 chunk
    for feat in set(_text_features(d["text"])):
        df[feat] += 1
```

**`set(...)` 是这段最容易看漏、却最关键的一步**:

```
假如 chunk1 里「报销」出现了 10 次:
  不加 set → df['报销'] 加 10  → ❌ 错
  加 set   → df['报销'] 只加 1  → ✅ 对
```

因为 df 统计的是"**多少个 chunk 包含这个词**",而不是"出现多少次"。一个 chunk 里说 100 遍「报销」和说 1 遍,对"区分不同文档"来说没有差别——说 100 遍也只代表这一个 chunk 有这个词。

> 这和数据库里的"去重计数"是同一个思路:`count(distinct chunk)` 而不是 `count(*)`。

### 第 228 行:算 IDF(整个函数的灵魂)

```python
self.idf = {f: math.log(n / (1 + c)) + 1.0 for f, c in df.items()}
#                    └┬┘  └──┬──┘ └──┬──┘
#                   chunk总数  +1平滑  +1保底
```

**直觉:这个词越"稀罕"(只在少数 chunk 出现)→ 越能代表某个 chunk 的主题 → 权重越高。**

- `n / (1 + c)`:c 越大(出现越普遍),这个比值越小
- `math.log(...)`:把比值压成对数量级(变化平缓)
- `+ 1.0`:保证权重最小也有 1,且 log 部分不会把权重压成负数

**为什么是 `1 + c` 而不是 `c`**:防极端情况(比如某些实现里 c 可能为 0 时除零),同时让权重变化更平滑。这是**加了平滑的 IDF 变体**(标准公式是 `log(N / df)`)。

### 第 230~232 行:向量化 + 存储

```python
for d in docs:
    self.chunks.append(d)                       # 存原文(检索后展示给 LLM 用)
    self.vectors.append(self._embed(d["text"])) # 存向量(内部用 idf 加权)
```

`self._embed`(第 207~216 行)做的事:把每个特征乘上它的 IDF 权重再累加并归一化——

```python
w = self.idf.get(feat, 1.5)   # 见过→用权重;没见过(查询里的生词)→给 1.5 中高权重
vec[i] += fv[i] * w           # 特征向量 × 权重,再累加
```

**关键词词向量被放大,高频虚词被缩小。** 这就是"模拟 TF-IDF"。

### 第 233~234 行:打印建库结果

```python
print(f"📥 建库完成:共 {len(self.chunks)} 个 chunk,"
      f"来源 {len({c['meta']['source'] for c in self.chunks})} 份文档")
```

`{c['meta']['source'] for c in self.chunks}` 是**集合推导式**,把所有 chunk 的来源文档名收进一个 set(自动去重),`len` 就是文档份数。一条 print 同时报了"多少 chunk、多少份文档"。

---

## 三、两个关键设计,面试必讲

### 1. `set(...)` 去重:数"多少个 chunk 包含",不数"出现多少次"

```python
for feat in set(_text_features(d["text"])):
```

这是文档频率的正确算法。忘了 `set`,高频词(比如一个 chunk 反复说「报销」)会被重复计数,IDF 权重全被带偏。

### 2. IDF 公式的平滑:让权重永远为正、且集中在 1 附近

```python
idf = log(n / (1 + c)) + 1.0
```

| 情况 | 代入(n=3) | 结果 |
|---|---|---|
| 词只在 1 个 chunk(c=1) | log(3/2)+1 | ≈ 1.41 |
| 词在 2 个 chunk(c=2) | log(3/3)+1 | = 1.00 |
| 词在 3 个 chunk(c=3) | log(3/4)+1 | ≈ 0.71 |

- 权重范围始终在 **0.7 ~ 1.4** 之间(小范围浮动,不会出现负数或爆炸)
- **出现越少 → 权重越高**,这符合直觉:稀罕的词更能代表文档主题
- 和标准公式 `log(N/df)` 的区别只是加了 `1+` 和 `+1.0` 两处平滑,目的都是防极端情况

---

## 四、完整例子(手算思路)

假设 `_chunk()` 切出了 3 个 chunk:

```python
docs = [
    {"text": "出差住宿标准:一线城市一晚不超过500元", "meta": {"source": "差旅规定.docx", "chunk": 0}},
    {"text": "差旅报销:打车费、交通费、住宿费、餐费均可报销", "meta": {"source": "报销指南.pdf", "chunk": 1}},
    {"text": "公积金提取:租房可每季度提取一次", "meta": {"source": "公积金政策.pdf", "chunk": 2}},
]
n = 3
```

**① 统计 df**(只看几个关键词):

```python
df["住宿"] = 2   # chunk0、chunk1 都包含
df["报销"] = 1   # 只有 chunk1
df["公积"] = 1   # 只有 chunk2
```

**② 算 IDF:**

```python
self.idf["住宿"] = log(3/(1+2)) + 1 = log(1)  + 1 = 1.00
self.idf["报销"] = log(3/(1+1)) + 1 = log(1.5)+ 1 ≈ 1.41
self.idf["公积"] = log(3/(1+1)) + 1 = log(1.5)+ 1 ≈ 1.41
```

结论:「住宿」两个 chunk 都有 → 区分度低 → 权重 1.0;「报销」「公积」各只属于一个 chunk → 关键词 → 权重 1.41。

**③ 存储:**

```python
self.chunks  = [chunk0, chunk1, chunk2]      # 原文,按序
self.vectors = [向量0, 向量1, 向量2]          # 与 chunks 下标一一对应
```

---

## 五、大白话:IDF 在干什么(手把手)

把每个词想象成"问题里的线索":

- 「报销」「公积」这类词:**很少出现,一旦出现就很有信息量** → 像"指纹",权重高
- 「住宿」这类词:**好几个文档都提** → 线索不够独特,权重中等
- 「的」「了」这类虚词:**哪都有** → 没有任何区分度 → 权重最低

建库时把这份"词 → 重要程度"的对照表(idf)算好存起来,之后:
- 用户问题里出现「报销」→ 它的向量分量被放大 → 和含「报销」的 chunk 撞得更狠 → 相似度更高
- 问题里出现「的」→ 分量被缩小 → 几乎不影响结果

**一句话:IDF = 给词配了"音量",越稀罕的词音量越大。**

---

## 六、和检索的关联(为什么权重重要)

```
用户问:「出差住宿一晚能报多少钱?」
  ↓ _embed 用同一份 idf 把问题向量化
    「住宿」×1.0、「报」×?(停用词,权重低)……
  ↓ retrieve 算问题向量和每个 chunk 向量的余弦相似度,取 Top-3
    chunk0(差旅规定)→ 有「住宿」→ 分高 → 排前面 ✅
  ↓ 交给 MockLLM 按关键词覆盖度挑选回答
```

**没有 IDF 会怎样?** 虚词(的/了/能)和所有 chunk 都撞车,相似度被"废话词"主导,检索结果就会跑偏。有了 IDF,关键词说了算。

---

## 七、面试一句话总结

> "`ingest` 是 RAG 的离线建库:先遍历所有 chunk,用 `Counter` 统计每个特征出现在多少个 chunk(文档频率,注意用 `set` 去重防止单 chunk 重复计数),再按 `log(n/(1+df))+1` 算 IDF——出现越稀罕的词权重越高;最后把每个 chunk 的原文和用 IDF 加权的向量存进库,检索时问题和文档用同一套权重向量化,让低频关键词主导相似度。它等价于真实项目里 embedding 模型的建索引过程,是 RAG 能准确检索的前提。"
