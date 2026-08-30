# Skill 路由第 ② 步 · Function Calling 精排怎么选 Skill + 生成参数

> 承接 [Skill路由决策原理](./Skill路由决策原理-Trae内部怎么帮你选Skill.md) 的 3 步总览,这篇只抠第 ② 步这一个点——
> 「RAG 给了 3 个候选 Skill id 之后,LLM 是怎么从 3 个里面挑 1 个,并且生成合法参数的?为什么必须走原生 Function Calling 而不是让模型自由选?这一步常见哪些坑,怎么填?」

---

## 一、一句话理解

第 ② 步做了三件事:**① 把候选 Skill「翻译」成 LLM 认识的工具说明书 → ② 模型在说明书里挑一本 → ③ 按说明书里的「下单要求」填好参数表**。

类比你点外卖的「下单页」:RAG 把「饺子」这个词从 100 家店筛成了 3 家给你看,然后你在 3 家的菜单里挑 1 家、再按菜单里的规格(大/中/小份、口味、加不加菜)填好提交——**挑店 + 填规格**两件事合起来,就是第 ② 步 Function Calling 精排在干的活。

---

## 二、代码级流程:1 个请求 + 1 个响应,2 张报文就讲透

### 2.1 输入:发给 LLM 的请求长什么样

RAG 粗筛返回了 3 个候选 id:
```
["table-page-generator", "form-page-generator", "chart-page-generator"]
```

Trae 内部按 id 从注册表拿每个 Skill 的 SkillMeta(见 SKILL.md 解析产物),然后调 `toTools()` 转成标准 Function Calling 声明:

```typescript
// 伪代码:toTools(candidatesOnly) —— 注意只转候选那 3 个,不是全库
function toTools(skillIds: string[]): Tool[] {
  return skillIds.map(id => {
    const meta = registry.get(id)!   // id 来自 RAG,一定存在
    return {
      type: 'function',
      function: {
        name: meta.name,                              // Skill 的唯一 id
        description: meta.description,                 // SKILL.md 的 description
        parameters: meta.parameters as JSONSchema,    // SKILL.md 的 parameters Schema
      }
    }
  })
}
```

最终 **LLM 实际收到的 HTTP 请求体**(关键是多了 `tools` 字段):

```json
POST /v1/chat/completions
{
  "messages": [
    { "role": "system",    "content": "你是一个前端开发助手,有工具就用工具。" },
    { "role": "user",      "content": "帮我做一个有筛选、有分页的用户管理表格页" }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "table-page-generator",
        "description": "生成前端表格页组件,支持筛选、分页、排序、自定义列配置、接口对接。不适用表单提交页和图表展示页。",
        "parameters": {
          "type": "object",
          "properties": {
            "page_title":       { "type": "string",  "description": "页面标题,例如用户管理、订单列表" },
            "columns":          { "type": "array",   "description": "列配置数组,每项={title, dataIndex, type, sorter}" },
            "enable_filter":    { "type": "boolean", "description": "是否启用行内条件筛选" },
            "enable_pagination":{ "type": "boolean", "description": "是否启用底部分页器" }
          },
          "required": ["page_title", "columns"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "form-page-generator",
        "description": "生成表单提交页组件,支持输入框/下拉/日期选择/校验规则。不适用表格列表页。",
        "parameters": { /* ...略 */ }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "chart-page-generator",
        "description": "生成图表展示页组件,支持折线图/柱状图/饼图/ECharts/D3。不适用纯数据列表。",
        "parameters": { /* ...略 */ }
      }
    }
  ],
  "tool_choice": "auto"
}
```

**关键点 3 个(面试追问素材)**

| 细节 | 为什么这么写 |
|---|---|
| `tools` 里**只有候选的 3 个**,不是全量 Skill | RAG 粗筛的价值就在这里:token 省 90%+ + 降低模型「选错函数」的概率(工具越多越混淆) |
| 每个 `parameters` 里每个字段都写了 `description` 带**示例** | Schema 里的 description 也是模型读的「参数说明书」,写含糊了参数就容易瞎填 |
| `description` 里**加了一句负面描述**(「不适用 XX 页」) | 边界防御,模型看到「不适用表单页」就会少犯把表格请求派去表单 Skill 的错 |

### 2.2 输出:LLM 返回的响应长什么样

模型看到上面的请求,原生 Function Calling 能力返回结构化 tool_call:

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": null,                                    // ← 注意是空的,没直接回答
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "table-page-generator",              // ← 选了哪个 Skill
              "arguments": "{\n  \"page_title\": \"用户管理\",\n  \"columns\": [\n    {\"title\":\"ID\",\"dataIndex\":\"id\",\"type\":\"number\"},\n    {\"title\":\"用户名\",\"dataIndex\":\"username\",\"type\":\"string\"}\n  ],\n  \"enable_filter\": true,\n  \"enable_pagination\": true\n}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"                         // ← 完成原因是 tool_calls
    }
  ]
}
```

**怎么看模型的决策:**
- `finish_reason = "tool_calls"` → 模型决定要调用工具,不直接回答;
- `tool_calls[0].function.name` → **选了哪个 Skill**(路由结果);
- `tool_calls[0].function.arguments` → **传给 Skill 的参数(JSON 字符串,要 parse)**;
- 如果 3 个都不合适,模型会返回 `finish_reason="stop"` + `content` 有文本,`tool_calls` 为空 → 代表「不调 Skill,直接用通用能力回答」(兜底)。

---

## 三、模型内部是怎么「选」的?(理解原理,不是死记)

Function Calling 能力是大模型训练阶段**专门训出来的**,不是 prompt 工程。训练时,模型见过几百万对「用户问题 + 正确的 {name, arguments}」,所以它学到了两个模式:

### 模式 A:选 name——看 description 和用户问题的语义匹配度

```
用户问题:「做一个用户管理表格页,要有筛选和分页」
tool 1 description:「生成前端表格页组件,支持筛选、分页...」→ 关键词都命中,语义最贴
tool 2 description:「生成表单提交页...」                    → 完全不贴
tool 3 description:「生成图表展示页...」                    → 完全不贴

→ name = "table-page-generator"
```

这一步**模型看不到 RAG 的分数**,纯靠读 description。所以命中率要提升,description 写得好比向量化模型好更重要(呼应策略 02/04)。

### 模式 B:生成 arguments——按 parameters Schema 填结构,值从用户问题里抠

```
Schema 要求:
  page_title (string, 必填)   → 从「用户管理表格页」里抠出 page_title = "用户管理"
  columns (array, 必填)      → 结合通用理解,默认给两列:ID 列 + 用户名列
  enable_filter (boolean)    → 从「要有筛选」里推断 true
  enable_pagination (boolean)→ 从「要有分页」里推断 true
```

**坑的高发点**(面试要能讲):

| 坑 | 为什么会发生 | 解法 |
|---|---|---|
| 必填字段没传 | Schema 里 required 写了,但模型忘了填 | Router 层 JSON Schema 校验失败 → 把错误格式化回喂模型重试 |
| 字段类型错(数字传成字符串) | Schema 类型标了但模型没严格遵守 | 同上,Schema 校验 + 回喂 |
| columns 这种数组结构瞎编 | 用户没给列,模型只能默认值,默认值可能错 | 在描述里写「columns 若未提供,请先用 get_project_file_structure 工具查现有组件库约定后再生成」→ 让模型主动先摸业务再填(这是 Skill 写得高级的信号) |

---

## 四、为什么必须走原生 Function Calling,不能让模型在 Prompt 里自由选?

这是面试高频追问(答不好会丢架构分)。对比两种实现:

| 维度 | 让 LLM 在 Prompt 里填 JSON | 原生 Function Calling(推荐) |
|---|---|---|
| **准确率** | 低,经常少引号、多逗号、字段名错 | 极高,模型训练过输出严格结构 |
| **参数 Schema 约束** | 纯靠 Prompt 文字描述,模型容易忽略 | 底层有约束,arguments 基本符合 JSON Schema 形状 |
| **多工具并行** | 非常难让模型同时输出多个调用 | 原生支持 `tool_calls` 数组,并行无依赖工具 |
| **可观测性** | 需要自己解析回答字符串里的 JSON | finish_reason/tool_calls 是结构化字段,日志/监控好做 |
| **安全** | 模型可能发明不在候选里的函数名 | tools 白名单就是候选集合,不可能跳出 3 个之外 |

一句话:FC 不是「省事写法」,是**模型训练就支持的结构化能力**——不用它等于放着官方给的高精度实现不用,自己在 prompt 里重造轮子,结果一定更差。

---

## 五、完整调用闭环:第 ② 步返回后,下一步怎么走(衔接 Router)

Router 拿到 `{name, arguments}` 之后做三件事:

```
① 解析 arguments:JSON.parse(arguments_str) 转成对象
   └─ 解析失败 → 返回解析错误给模型重出

② 校验:
   a) registry.has(name)            → name 必须在注册表白名单(防幻觉)
   b) jsonschema.validate(args, schema) → 参数类型/必填/枚举对不对
   c) permissions 授权校验         → 高危操作弹用户确认
   └─ 任何一项失败 → 把错误信息(注意格式化,不是堆栈!)拼成 tool_result 回喂模型,
                     让模型自纠错后重试,最多 N 次再放弃

③ 都通过 → 查注册表拿 path → 交给 Executor 执行 Skill
   └─ 执行结果再作为 tool_result 回喂 → 回到 Function Calling 或 ReAct 循环
```

这也是你简历③写的「模型解析—函数执行—结果回填闭环」这句话的代码级含义。

---

## 六、这一步常见的真实坑(讲出来就是有实操经验)

### 坑 1:参数没填全,直接报错给用户
❌ 错误:Router 校验失败直接 `throw new Error` 给前端。
✅ 正确:把错误**格式化**后回喂模型重试:
```
tool_result = {
  success: false,
  error_code: "VALIDATION_ERROR",
  message: "调用 table-page-generator 失败:缺少必填字段 'columns',请补充列配置后重试调用"
}
```
模型看到这条会自己修正参数再调,用户完全无感。这是「结果回填闭环」的核心——FC 不是一次性的。

### 坑 2:description 里没写边界,两个 Skill 互相抢流量
表格 Skill 和表单 Skill 的 description 都只写正面不说反面,用户说「做个提交页面」时,表格 Skill 里有「按钮」的词,相似度反而高,被 RAG 排第一,LLM 就选错。
**解法**:每个 description 尾部必须加 1 句负面描述:「不适用:表单提交页 / 纯数据列表」——这就够把误选率砍一半以上。

### 坑 3:Skill 太多,RAG 粗筛漏了正确的,第 ② 步再厉害也没用
第 ② 步的候选只来自第 ① 步,RAG 没召回进来的 Skill,LLM 连看都看不到。
**解法**:命中率问题要**先查召回率再查精排率**——打日志:RAG 结果里有没有命中黄金 Skill?没有先修 ① 步描述,有但 LLM 不选才修 ② 步 Schema/描述。

### 坑 4:arguments 里是字符串不是对象,代码直接取字段报错
`tool_calls[0].function.arguments` 返回的一定是字符串(模型输出的是文本),不要漏 `JSON.parse`。

---

## 七、面试一句话版本(背这段)

> Function Calling 精排分两步:先把 RAG 召回的 3~5 个候选 Skill,从注册表取 SkillMeta,用 `toTools()` 转成原生 FC 的 tools 声明格式(name/description/parameters Schema),只带候选不带全量,塞到请求体 `tools` 字段里发给模型;模型用训练过的 FC 能力做两件事——① 读 3 份 description 语义匹配,选一个 name(不可能跳出 tools 白名单);② 按 parameters Schema 的字段定义,从用户问题里抠值填出结构化 arguments。返回 `finish_reason="tool_calls"` + `tool_calls=[{name, arguments}]`,都不合适就走兜底不调用。Router 拿到后先 JSON.parse arguments,再做三件校验(name 在注册表白名单里、参数过 JSON Schema、权限授予),全通过后查路径执行,任何失败都把格式化错误回喂模型自纠错重试,最多 N 次再放弃。
>
> 为什么走原生 FC 而不是 prompt 里自由选:因为 FC 是模型训练就支持的结构化能力——准确率高、参数约束强、天然多工具并行、带白名单安全,自己 prompt 重写轮做不到。

---

## 八、关联笔记索引

- [06-内部实现:SKILL.md 如何变成 tools](./06-内部实现-SKILL.md如何变成tools.md)(本文 2.1 的 `toTools()` 伪代码和完整 SkillMeta 接口的原文出处)
- [FunctionCalling/知识点.md](../../FunctionCalling/知识点.md)(FC 5 步总流程图 + Schema 写法 + 错误回填机制通用版)
- [Skill路由决策原理-Trae内部怎么帮你选Skill](./Skill路由决策原理-Trae内部怎么帮你选Skill.md)(第 ② 步放在 3 步整体链路里的位置和上下游衔接)
- [02-策略01-描述优化写何时用](../02-策略01-描述优化写何时用/知识点.md)(description 怎么写,直接影响这一步的 name 选择)
- [04-策略03-边界防御负面描述](../04-策略03-边界防御负面描述/知识点.md)(负面描述怎么写,直接解决 name 选错的高频问题)
