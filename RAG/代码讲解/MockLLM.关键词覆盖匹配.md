# `MockLLM` 关键词覆盖匹配解析:挑出和问题最相关的材料

> 出处:`真实例子.py` 第 180~186 行,`MockLLM.answer` 的核心逻辑。

```python
# 模拟 LLM 逐条「阅读」检索到的材料:找到覆盖了关键词最多的那条
best, best_hit = 0, None
for h in hits:
    hit_words = set(_text_features(h["text"]))
    overlap = sum(1 for w in q_keywords if w in hit_words)
    if overlap > best:
        best, best_hit = overlap, h
```

---

## 一、这段代码在干嘛(一句话)

> **遍历检索到的候选材料(`hits`),数每条材料和问题的关键词重合几个,挑出重合最多的那一条**。重合多的 = 和问题最相关 = 用它来回答。

这是对真实 LLM"阅读材料 → 判断哪段能回答问题"的**规则化模拟**。

---

## 二、逐行拆解

### 0. 前提:两个"词表"已经备好

```python
q_keywords = [f for f in _text_features(question) if len(f) >= 2 and f not in self._STOPWORDS]
```

- `q_keywords`:**问题的关键词列表**(双字词组,去掉"的、了、吗"等停用词)
- 例:问"**出差打车费能报销吗**" → 关键词可能是 `["出差", "打车", "报销"]`

### 1. `for h in hits:`——遍历每条候选材料

`hits` 是向量检索返回的 Top-K 候选文档块,每条长这样:

```python
{"text": "差旅报销:出差产生的打车费、交通费...", "meta": {"source": "报销制度.pdf", "chunk": 1}}
```

### 2. `set(_text_features(h["text"]))`——材料特征词去重

```python
hit_words = set(_text_features(h["text"]))
```

- `_text_features(h["text"])` 把材料文本拆成特征词(单字 + 双字)
- `set(...)` 转成**集合**,两个目的:
  1. **去重**:同一词只算一次,避免一条材料里"报销"出现 5 次就虚高
  2. **查得快**:`w in hit_words` 从遍历整个列表变成 O(1) 哈希查找

### 3. `overlap = sum(1 for w in q_keywords if w in hit_words)`——数重合关键词

```python
sum(1 for w in q_keywords if w in hit_words)
```

**数一数:问题的关键词里,有几个出现在这条材料里?**

- `w in hit_words` → 这个词材料里有吗?有就出 1,没有就出 0
- `sum(...)` → 把所有 1 加起来,就是重合数

```python
# 例:问题关键词 ["出差", "打车", "报销"]
# 材料特征词 {"差旅", "出差", "打车", "交通", "住宿", ...}
# 重合:"出差"✓ "打车"✓ "报销"✗ → overlap = 2
```

### 4. `if overlap > best:`——记下当前最大

```python
if overlap > best:
    best, best_hit = overlap, h
```

- `best` 记录目前见过的**最大重合数**
- `best_hit` 记录对应**那条材料**
- 循环结束后,`best_hit` 就是所有候选里和问题最相关的一条

用 `>` 而非 `>=`:第一条候选也能入选(0 > 0 为假,但第一条如果 overlap 是 2 就 > 0 成立),并且并列时保留先遇到的。

---

## 三、完整例子走一遍

假设检索回 3 条候选,问题是"出差打车费能报销吗"(`q_keywords = ["出差", "打车", "报销"]`):

| 候选材料 | 材料特征词里含问题关键词 | overlap |
|---|---|---|
| 差旅报销:出差产生的打车费、交通费... | 出差、打车 | **2** |
| 工资发放:每月 15 日发放上月工资... | 无 | 0 |
| 加班政策:工作日加班按 1.5 倍工资... | 无 | 0 |

循环结果:`best = 2`,`best_hit = 差旅报销那条`。

接着第 189 行:

```python
if best >= 2 and best_hit is not None:
    return f"根据《{best_hit['meta']['source']}》规定:{best_hit['text']}"
```

重合 ≥ 2 才算"相关",返回:👆

```python
根据《报销制度.pdf》规定:差旅报销:出差产生的打车费、交通费、住宿费、餐费均可报销,需附行程单和发票。
```

`best >= 2` 就是防幻觉的**置信度门槛**:只重合 0~1 个词,宁可承认不知道。

---

## 四、为什么用 `set`(面试小考点)

```python
hit_words = set(_text_features(h["text"]))
```

| | 列表 list | 集合 set |
|---|---|---|
| `w in hit_words` 速度 | O(n) 逐个比 | O(1) 哈希 |
| 重复词 | 会重复计数 | 自动去重 |

一条材料里"报销"出现 3 次,用列表会让 overlap 虚增到 3,用 set 老老实实算 1 次。**"数关键词重合"天然需要去重 + 快速查找,set 是正确选择。**

---

## 五、和真实 LLM 的区别

- **这里**:数"词面重合数"——规则化、可解释,但看不懂同义词
- **真实场景**:LLM 是语义理解——"打车费"和"交通费"也能算相关(代码 118 行注释专门提到这点)

所以这只是**演示用的替代品**,真实 RAG 里这段会被换成:`response = llm.chat(prompt + 相关材料)`。

---

## 六、面试一句话总结

> "这段代码模拟 LLM 从检索候选中挑选最相关材料:把每条候选的特征词转成 `set` 去重,统计与问题关键词的重合数 `overlap`,`if overlap > best` 保留重合最多的一条。最后用 `best >= 2` 做置信度门槛,相关才回答、否则承认不知道——这是 RAG 防幻觉的关键设计,真实项目里这一步由 LLM 的语义理解完成。"
