# Skill 路由决策原理 · Trae 内部怎么帮你选 Skill

> 回答的问题:「我在 Trae 里说了一句话,系统到底靠什么决定调用哪一个 Skill?为什么有时候选不对?选 Skill 和 RAG、Function Calling 什么关系?」
>
> 结论先说:选 Skill **不是「向量分数最高就直接用」**,而是三段接力:**RAG 粗筛 Top-3 候选 → LLM Function Calling 精排生成 {name, args} → Router 校验查路径执行**,任何一段出问题都会导致命中率下降。

---

## 一、总览:3 步决策图(先记住这个结构)

以用户说「**帮我做一个有筛选、分页的用户管理表格页**」为例:

```
用户输入
   │
   ▼ ① RAG 粗筛(向量 + 关键词 混合检索)
   │  输入:用户原始需求
   │  做:embedding → 和所有 Skill description 算余弦相似度 + 关键词倒排
   │  输出:Top-3 候选 skill id + 分数
   │        [{id: "table-page-generator",  score: 0.92},
   │         {id: "form-page-generator",   score: 0.45},
   │         {id: "chart-page-generator",  score: 0.41}]
   │  谁说了算:RAG(只看语义相似度,不理解 Skill 的真实能力边界)
   │
   ▼ ② LLM Function Calling 精排⭐(真正决定选哪一个)
   │  输入:候选 Skill 转成的 tools 声明(每个 = name + description + parameters Schema)
   │  做:LLM 原生 FC 能力在 tools 里选一个,同时按 Schema 生成参数
   │  输出:{ name: "table-page-generator",
   │          arguments: {"page_title": "用户管理", "columns": [...], ...} }
   │  谁说了算:LLM(能理解 description 语义 + 参数匹配度,能判断「都不合适就不调工具」)
   │
   ▼ ③ Router 安全兜底 + 查表执行
   │  做 A:name 是否在注册表白名单里(防幻觉 name)
   │  做 B:arguments 按 JSON Schema 校验类型/必填(防参数错)
   │  做 C:权限校验(Skill 声明的 permissions 是否已授权,未授权弹窗确认)
   │  做 D:查注册表:registry.get(name).path → 拿到 skills/xxx/ 目录
   │  输出:按 path 加载 SKILL.md + 模板,进入执行流程
   │  谁说了算:Trae 代码(确定性逻辑,哈希查表 O(1),永不乱)
   │
   ▼
执行结果(生成表格页代码)
```

---

## 二、每一步拆细讲:为什么这么设计,怎么就会选不对

### 第 ① 步:RAG 粗筛——类比 HR 看简历挑候选人

**怎么做的**:

1. **启动时(一次性)** :扫描所有 Skill,读每个 SKILL.md 的 description → embedding 成向量 → 塞进向量库(存 `skill_id ↔ embedding_vector`);同时分词建关键词倒排表 `词 → [skill_id]`;
2. **运行时(每次提问)** :用户提问 embedding 成 `query_vec` → 向量库 Top-K 余弦相似度 + 关键词倒排命中 → 两路合并去重、分数加权 → 只留前 3~5 个候选 id。

**RAG 只干一件事:「缩小范围」,不承担最终决策**。它的短板很明确:

- 相似度高 ≠ 真能用:用户说「生成一份表格」,可能要的是 Markdown 表格(不是 React 组件),语义上确实像但 Skill 做不了;
- 相似度低 ≠ 不该选:用户用了冷门说法(如「列表视图页面」= 表格页),只要 description 没写这个词,RAG 就可能把它排到后面。

**这一步命中率出问题时**(你笔记里对应的策略):

| 现象 | 根因 | 对应策略 |
|---|---|---|
| 正确的 Skill 根本没进 Top-3(召回失败) | description 里没写用户会说的关键词/同义词 | 策略 02:描述优化写清「何时用」,补同义词覆盖 |
| 两个 Skill 描述差不多,常排错名次 | 边界没写,相似度互相干扰 | 策略 04:边界防御,负面描述写清「什么时候不用这个 Skill」 |

---

### 第 ② 步:LLM Function Calling 精排——类比技术面试官定人+定入职要求

**怎么做的**:

```
① 按第 ① 步返回的候选 id,从注册表取出每个 Skill 的 SkillMeta(name/description/parameters)
② 用 registry.toTools() 转成 LLM 认识的 FC 声明格式:
   [ {"type":"function","function":{"name":...,"description":...,"parameters":...}}, ... ]
③ 塞到请求体的 tools 字段里,和用户消息一起发给 LLM
④ LLM 返回结构化 tool_call:{"name": "...", "arguments": "{...}"}
   (如果 3 个都不合适,LLM 会返回「不调用任何工具」,相当于不选 Skill)
```

**这一步才是真正的选 Skill,和 RAG 完全独立**——LLM 根本看不到第 ① 步的分数,它只看到 3 份 tools 说明书,自己判断哪个最匹配。

**设计上的两个关键点(面试加分)**:

1. **为什么不把所有 Skill 都直接塞给 LLM?**
   两个原因:① Skill 多了会挤爆上下文 token;② 工具越多,模型选错的概率越高(「工具选择混淆」)。所以先用 RAG 砍到 3~5 个,再交给 LLM 精排。
2. **为什么必须转成原生 Function Calling 的 tools 声明格式,而不是让 LLM 读描述自由选?**
   - 模型原生 FC 能力是训练过的,选函数 + 生成 JSON 参数的准确率远高于「让它在 prompt 里填单选框」;
   - 参数 Schema 直接约束了输出结构,Router 层能做确定性校验;
   - 结果可预测:模型不可能凭空发明一个不被 tools 声明包的 name。

**这一步命中率出问题时**:

| 现象 | 根因 | 对应策略 |
|---|---|---|
| 进了候选但被 LLM 选了另一个错的 | description 边界模糊,或负面描述没写 | 策略 04:负面描述里写「不适用场景」 |
| 参数总生成错(漏必填/类型错) | parameters Schema 写得含糊,description 没给示例 | Function Calling 知识点:Schema 里每个字段都要写清含义和示例 |
| 都不合适硬选了一个 | LLM 没学会「说不会」,description 里缺「兜底/澄清信号」 | 策略 06:兜底机制——模糊场景触发反问澄清 |

---

### 第 ③ 步:Router 兜底 + 查表——类比人事系统验门禁 + 安排工位

**怎么做的(4 件事,全是你代码做的,模型不参与)**:

```
3A. name 白名单校验:registry.has(name)
    ├─ 是 → 继续
    └─ 否 → 模型幻觉(比如编了个不存在的 Skill 名),拒绝,让模型重选

3B. arguments Schema 校验:jsonschema.validate(arguments, parameters)
    ├─ 通过 → 继续
    └─ 失败 → 把具体错误(「columns 必填」「page 应为数字」)格式化回喂 LLM,
             要求它修正参数后重试(不是直接报错给用户!)

3C. 权限校验:SkillMeta.permissions 里声明的权限是否已授予
    ├─ 是 → 继续
    └─ 否 → IDE 弹窗让用户确认(写文件/执行终端这种高危操作必须经用户手)

3D. 查路径:path = registry.get(name).path → 拿到真实文件目录
    → 按 path 加载 SKILL.md + templates/scripts → 交给 Executor 执行
```

**安全红线(面试背这句)**:LLM 从头到尾只看到 `id + description + parameters`,**永远接触不到真实文件路径**。路径只在第 3D 步由查表揭晓——这是确定性逻辑,哈希查找 O(1),既稳又防 Prompt 注入越权。

---

## 三、最容易混淆的关系:选 Skill = RAG 吗?

**答案:用到了 RAG,但选 Skill 这件事本身 ≠ RAG。**

对照表把它们拆开:

| | RAG(知识库问答) | Skill 路由里的 RAG(第①步) | Skill 路由整体(三步加起来) |
|---|---|---|---|
| **检索对象** | 文档 chunk 原文 | Skill description 文本 | Skill(工具) |
| **返回物** | chunk 正文(直接喂 LLM 让它基于原文回答) | 只返回 skill id(后面查表拿描述) | 执行结果:代码/文档/页面等 |
| **有 Function Calling 参数生成吗?** | 没有,RAG 不生成参数 | 没有,RAG 只召回 id | 有,第 ② 步 FC 精排做 |
| **有注册表/路径安全映射吗?** | 没有,原文直接给用户 | 没有,RAG 不碰路径 | 有,第 ③ 步 Router 做 |
| **有工具执行权限校验吗?** | 没有,RAG 不执行 | 没有,RAG 不执行 | 有,第 ③ 步做 |

通俗对比:
- 知识库 RAG = **图书馆查书**,查完把书给你,你自己读;
- Skill 路由的 RAG 粗筛 = **公司里帮你找「哪个部门管这件事」**,给你 3 个候选部门名;
- Skill 路由整体(三步)= **找部门 + 找部门里对的人 + 让他做 + 验他有没有权限做 + 他真的把事做完给你**。

所以:选 Skill 用到了 RAG 作为「粗筛工具」,但整套机制是 **RAG + Function Calling + Registry 查表** 三者拼接出来的 Agent 工具路由系统,远大于 RAG 本身。

---

## 四、常见问题(面试会被问到的)

### Q1:为什么不直接「相似度最高就用谁」?

因为相似度高只代表「description 里的词像」,不代表「这个 Skill 真实能力对得上用户需求」。

典型失败案例:用户要「生成一页包含表格 + 表单的用户详情页」→ 单表格 Skill 描述分数最高,但它不支持混合页面,选了就会生成错误代码。必须靠 LLM 读完整 description + parameters Schema 综合判断,或在都不对时返回「都不合适」走兜底。

### Q2:Skill 数量多了,第一步 RAG 召回效果下降怎么办?

三招,按优先级:① Skill 去重合并(同类只留一个正主,避免互相抢分数);② 分层:先做「Skill 组」粗召回(如「前端生成类/测试类/运维类」),再在组内召回具体 Skill(两级 RAG);③ Metadata 过滤:按场景标签(前端/后端/测试)先圈定范围,再算相似度。

### Q3:第 ② 步 LLM 返回「都不调用工具」怎么处理?

两种策略:

- **默认策略**:降级为通用 LLM 直接回答(不用任何 Skill)。这是兜底,用户请求本来就不需要工具时就是正常行为;
- **高召回场景策略(如 Agent 工具库)**:加一轮澄清提问——「我有以下几个工具可能帮上忙:A 做 XX、B 做 XX、C 做 XX,你想走哪个,还是想描述更细一点?」,这也是你笔记里策略 06 的兜底机制。

### Q4:怎么证明 Skill 路由真的用到了 RAG?

不需要反编译 Trae,从行为就能推出来:

1. Skill 丢进去就能用,你没写任何注册代码 → Trae 启动时必然做了扫目录+建索引;
2. 说「帮我做个列表页带筛选」能匹配到「表格页生成器」(同义词匹配成功,非单纯关键词)→ 必然做了语义向量化;
3. 换个更像的 Skill 进去,同一问题选的 Skill 会变 → 排序是靠相似度动态决定的。

### Q5:用户自定义 Skill 和 Trae 内置 Skill 是一起路由吗?优先级怎么定?

按你笔记 06 篇,启动时扫描三个位置(内置 → 用户级 → 项目级),全部进同一个注册表统一路由。**id 冲突时,项目级覆盖用户级覆盖内置**(「就近优先」原则),方便团队 override 官方 Skill。路由本身不带权重逻辑,纯看相似度。

---

## 五、面试一句话版本(背这段)

> Trae 选 Skill 分三段接力:第一段是 **RAG 粗筛**——用户提问转向量,和所有 Skill 的 description 算余弦相似度 + 关键词倒排合并,Top-3 出候选 id;第二段是 **Function Calling 精排**——把候选 Skill 转成 FC 的 tools 声明(name/description/parameters Schema)塞给模型,LLM 在候选里选一个并生成符合 Schema 的参数;第三段是 **Router 安全兜底**——name 查注册表白名单、arguments 走 JSON Schema 校验、权限校验,全通过后才按 name 查表拿路径执行。选 Skill 不是「向量分数最高就直接用」,RAG 只负责缩小范围,最终决策是 LLM FC 做的,安全校验是确定性代码做的。三者分工:RAG 像 HR 筛简历,FC 像技术面试官定人定需求,Router 像人事查门禁配工位。

---

## 六、关联笔记索引

- `01-总题干-Skill命中率/知识点.md`(命中率三大支柱:描述优化/边界防御/兜底机制)
- `01-总题干-Skill命中率/Skill路由执行-注册表与id接力/02-索引与检索怎么组织.md`(本文第 ① 步的代码级细节:双索引结构/召回返回结构)
- `01-总题干-Skill命中率/Skill路由执行-注册表与id接力/06-内部实现-SKILL.md如何变成tools.md`(本文第 ②③ 步的代码级:SkillMeta 接口/toTools() 转换/FC 请求结构/Router 校验 4 件事)
- `01-总题干-Skill命中率/Skill路由执行-注册表与id接力/05-职责分工与安全边界.md`(本文第 ③ 步的安全设计:LLM 永远不碰路径的原因)
