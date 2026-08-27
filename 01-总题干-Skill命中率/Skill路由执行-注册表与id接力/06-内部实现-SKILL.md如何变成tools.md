# 06 内部实现:SKILL.md 如何变成 tools

> 上一篇:[05-职责分工与安全边界](./05-职责分工与安全边界.md) | 返回:[README](./README.md)
>
> 一句话:**注册 = 解析 SKILL.md 进内存 Map;调用 = 转成 function calling 的 tools 声明塞给 LLM,LLM 返回 `{name, arguments}`,Executor 照着执行。**

---

## 一、全景:3 步把"文件"变成"工具"

```
SKILL.md ──① Loader 解析──→ SkillMeta(内存对象)
         ──② Registry 注册──→ Map<name, SkillMeta>
         ──③ Router 转 tools──→ function calling 声明
                                   │
                       塞进 LLM 请求的 tools 字段
                                   │
                              LLM 输出 {name, arguments}
                                   │
                    校验 schema + 查权限 → Executor 真执行
                                   │
                          stdout/stderr 回喂 → Agent 循环
```

> 核心洞察:SKILL.md 本身 LLM **看不到**;LLM 看到的是 Registry 转出来的 **tools 声明**。这就是"注册"的实质。

---

## 二、Loader:扫目录 + 解析 frontmatter

**扫哪三个位置(按优先级):**

| 位置 | 路径 | 生效范围 |
|---|---|---|
| 内置 Skill | `<安装目录>/builtin/skills/` | 平台自带,所有项目 |
| 用户级 Skill | `~/.trae/skills/`(全局配置目录) | 用户所有项目 |
| 项目级 Skill | `<你的项目>/.trae/skills/` | 仅当前项目 |

**解析器干什么:** 读 `SKILL.md` → 切出 YAML frontmatter → 解析成 `SkillMeta`:

```yaml
---
name: run_terminal_command        # 唯一 id,路由只认它
description: 执行终端命令并返回 stdout/stderr
permissions: [terminal.write]     # 权限声明,执行前校验
parameters:                       # 参数 schema(JSON Schema)
  type: object
  properties:
    command: { type: string }
  required: [command]
---
```

```ts
interface SkillMeta {
  name: string
  description: string
  permissions: string[]
  parameters: JSONSchema
  path: string
}
```

---

## 三、Registry:内存 Map + 一个关键方法 `toTools()`

```ts
class SkillRegistry {
  private skills = new Map<string, SkillMeta>()

  register(meta: SkillMeta) { this.skills.set(meta.name, meta) }

  get(name: string) { return this.skills.get(name) }   // O(1) 查路径/schema

  /** ★ 关键:把注册表转成 LLM 认识的 function calling 声明 */
  toTools(): Tool[] {
    return [...this.skills.values()].map(s => ({
      type: 'function',
      function: {
        name: s.name,
        description: s.description,
        parameters: s.parameters,   // schema 原样透传
      },
    }))
  }
}
```

**toTools() 的产物**(LLM 真正看到的):

```json
[
  {
    "type": "function",
    "function": {
      "name": "run_terminal_command",
      "description": "执行终端命令并返回 stdout/stderr",
      "parameters": {
        "type": "object",
        "properties": { "command": { "type": "string" } }
      }
    }
  }
]
```

---

## 四、Router:把 tools 塞给 LLM + 解析返回值

**请求结构:** 平时对话只发 `messages`,注册了 Skill 后多一个 `tools` 字段:

```
POST /v1/chat/completions
{
  messages: [ { role: "user", content: "帮我在项目里跑一下测试" } ],
  tools: [ ...toTools() 的产物... ]        ← 注册表在这亮相
}
```

**LLM 的返回(结构化工具调用,不是自然语言):**

```json
{
  "name": "run_terminal_command",
  "arguments": "{\"command\": \"npm test\"}"
}
```

**Router 拿到后做两件事:**
1. **查表**:`registry.get(name)`——不存在 → 报错/重选;
2. **验参**:`arguments` 按 `parameters` schema 校验(必填项、类型),不合格 → 要求 LLM 重出。

---

## 五、Executor:照着参数真执行

```
① 权限校验    name 对应的 permissions 是否已授权(未授权 → IDE 弹窗让用户确认)
② 真执行      开终端 / 写文件 / 跑代码搜索(按 Skill 类型)
③ 结果回喂    stdout/stderr → 当上下文喂回 LLM
④ Agent 循环  规划→行动→观察→验证→结论,直到任务完成
```

---

## 六、两个值得记的细节

1. **"注册"的全部本质 = 进 Map + 转 tools**:你导入 SKILL.md 那一刻,平台只是解析成 `SkillMeta` 放进内存 Map,并在下轮请求里以 tools 形式出现——没有更玄的东西;
2. **SKILL.md 是开放标准**:该格式(YAML frontmatter + Markdown)来自 Anthropic 的 Agent Skills 规范,Trae 等多平台遵循它,所以解析器逻辑通用,**换个平台 skill 文件能复用**。

---

## 七、面试一句话

> "用户零代码接入,是因为 loader/registry 在平台内部:启动时扫描内置/用户级/项目级三个目录,解析 SKILL.md 的 frontmatter 得到 SkillMeta,注册进内存 Map;每次请求时用 toTools() 把注册表转成 function calling 的 tools 声明塞给 LLM,LLM 输出 {name, arguments},Router 查表验参后交给 Executor 真执行。用户写的只是 SKILL.md,LLM 看到的只是 tools。"

---

## 八、对照表:六个文件怎么连成一条链

| 文件 | 回答的问题 |
|---|---|
| 01 | Skill 文件怎么摆?注册表存什么? |
| 02 | 召回时去哪查?索引怎么建? |
| 03 | 代码拆成哪些模块?怎么依赖? |
| 04 | 启动/运行时按什么顺序跑? |
| 05 | 谁负责什么?为什么 LLM 不碰路径? |
| **06(本文件)** | **SKILL.md 具体怎么变成 tools?代码级** |
| 07 | 完整例子:复杂筛选表格页生成器(总 Skill 内部怎么组织) |
