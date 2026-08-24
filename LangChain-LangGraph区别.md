# LangChain vs LangGraph：什么区别（面试必问）

> 面试高频题。一句话记忆：
> - **LangChain = 乐高套装 / 预制菜**：LLM 应用开发框架，组件丰富，快速搭标准场景
> - **LangGraph = 电路图 / 流程图**：有状态的 Agent 编排引擎，精细控制每一步、支持循环与暂停恢复
> - **关系**：LangGraph 是 LangChain 生态的一部分，官方推荐**复杂 Agent 用 LangGraph 构建**

---

## 一、一句话版（先背这个）

| 框架 | 是什么 | 定位 |
|---|---|---|
| **LangChain** | LLM 应用开发框架：把"模型、Prompt、工具、Retriever、记忆、输出解析"做成可复用的组件，用 `Chain` 串联 | **组件库 + 编排**，快速搭建 LLM 应用 |
| **LangGraph** | 面向 **Agent** 的底层编排引擎：用**图（Graph）**定义 节点/边/条件边/状态，支持循环、持久化、人工介入 | **有状态的 Agent 运行时**，精细控制执行流程 |

---

## 二、核心区别对比表（面试答这张表）

| 维度 | LangChain | LangGraph |
|---|---|---|
| **核心抽象** | Chain（链）：`A → B → C` 顺序拼接 | Graph（图）：节点 + 边，可分支、可循环 |
| **编程范式** | 声明式链式：`prompt \| model \| parser` | 显式定义 节点/边/状态机 |
| **状态管理** | 弱：状态靠链内传递，全局共享麻烦 | 强：显式 `State`（如 TypedDict），所有节点读写同一个状态 |
| **循环** | 不擅长（链本质是线性 DAG，循环靠 hack） | **原生支持**：`add_conditional_edges` 条件边来回跳转 |
| **可控性** | 黑盒：Agent 跑起来难精确控制每一步 | 白盒：每个节点、每次跳转都可控制、可打断 |
| **可观测/调试** | 相对弱 | 强：检查点（Checkpoint）、每一步状态可回放 |
| **持久化/断点续跑** | 无内置 | 内置 Checkpointer：崩溃恢复、Time Travel 回退重放 |
| **人工介入** | 麻烦 | 原生支持 Human-in-the-loop（人工审核后再继续） |
| **适用场景** | 简单线性任务：单轮问答、翻译、RAG 问答链 | 复杂 Agent：多步规划、循环调用工具、需要状态与人工确认 |
| **定位** | 高层框架（开发效率优先） | 底层引擎（控制力优先） |

> 记忆口诀：**LangChain 管"标准化步骤"，LangGraph 管"任意流程"**。
> 链（Chain）只能往前走，图（Graph）可以拐弯、绕圈、回头。

---

## 三、生活化类比

| 类比 | LangChain | LangGraph |
|---|---|---|
| 做饭 | 预制菜包：按包装说明一步步来，简单快 | 厨师自由发挥：看情况加料、试味、调整 |
| 出行 | 直达大巴：固定的 A→B→C 路线 | 实时导航：根据路况随时改道、绕路、回头 |
| 乐高 | 套装说明书：按图纸拼 | 自由拼搭：任意组合、拆了重拼 |
| 工作流 | 流水线：一道工序接一道 | 红绿灯系统：每个路口都能判断放行还是拦截 |

---

## 四、代码对比（一眼看懂）

### LangChain 写法：线性链（适合简单任务）

```python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# LCEL 链式：prompt → model → parser，一条直线
chain = (
    ChatPromptTemplate.from_template("把 {text} 翻译成英文")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

chain.invoke({"text": "你好"})   # "Hello"
```

**特点**：声明式、代码短、开箱即用。但流程固定，没有分支和循环。

### LangGraph 写法：图（适合复杂 Agent）

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):        # 1. 定义全局状态
    messages: list
    tool_calls: int

def call_model(state):              # 2. 节点：让 LLM 思考
    ...
    return {"messages": [response]}

def call_tool(state):               # 3. 节点：执行工具
    ...
    return {"messages": [result]}

def should_continue(state) -> str:  # 4. 条件边：决定下一步去哪
    return "tool" if state["tool_calls"] < 3 else "end"

# 5. 拼装成图
graph = StateGraph(AgentState)
graph.add_node("model", call_model)
graph.add_node("tool", call_tool)
graph.set_entry_point("model")
graph.add_conditional_edges("model", should_continue,
                            {"tool": "tool", "end": END})
graph.add_edge("tool", "model")     # tool → model 形成循环！
app = graph.compile()

app.invoke({"messages": [("user", "帮我查一下今天天气")]})
```

**特点**：状态显式、节点可复用、条件边支持分支和循环、`tool → model` 能绕圈直到任务完成。

> **本质差异**：LangChain 的链是"一条直线"，LangGraph 的图是"能拐弯、能绕圈、能回头的流程图"。

---

## 五、什么时候用哪个（选型指南）

| 场景 | 选谁 |
|---|---|
| 快速做 Demo、简单问答、翻译、单次 RAG | **LangChain** |
| 固定流程的 RAG 问答链（检索→增强→生成） | **LangChain** |
| 多工具 Agent：边想边调工具、循环直到完成 | **LangGraph** |
| 流程要分支：命中就答、没命中就调 API、再不行就拒答 | **LangGraph** |
| 需要持久化（崩溃续跑）、人工确认、步骤回放 | **LangGraph** |
| 既想要组件，又想要精确控制 | **LangGraph + LangChain 组件**（两者兼容） |

> 官方推荐：**复杂 Agent 优先用 LangGraph**。LangChain 的 AgentExecutor 已标记为 legacy（旧方案），新项目官方引导用 LangGraph（`create_react_agent` 等）。

---

## 六、两者怎么配合（重要！不是二选一）

LangGraph 是 **LangChain 生态的一部分**，两者是"上下层"关系：

```
            ┌────────────────────────────┐
            │        LangGraph           │  编排层：节点/边/状态/循环
            │   （Agent 的骨架与流程）      │
            └────────────┬───────────────┘
                         │ 节点内部调用
            ┌────────────▼───────────────┐
            │        LangChain           │  组件层：模型/Prompt/工具/Retriever
            │  （ChatOpenAI、Tool、检索器） │
            └────────────────────────────┘
```

- **LangGraph 的每个节点里**，可以直接用 LangChain 的组件（`ChatOpenAI`、`create_retriever_tool`、`OutputParser`）
- **LangChain 的组件**在 LangGraph 里照常工作，两者 API 同源

```python
# LangGraph 节点内部照用 LangChain 组件
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

model = ChatOpenAI(model="gpt-4o-mini")   # LangChain 组件

def agent_node(state):
    resp = model.invoke(state["messages"])   # 在 LangGraph 节点里调用
    return {"messages": resp}

# ... 其余同上面的图定义
```

> 类比：**LangGraph 是"生产线图纸"，LangChain 是"设备零件库"**。图纸决定流程怎么走，零件决定每一步干什么。

---

## 七、面试速答（背这段）

> **Q：LangChain 和 LangGraph 什么区别？**
> LangChain 是 LLM 应用开发框架，提供模型、Prompt、工具、Retriever 等组件，用链式（Chain）把标准流程串起来，适合简单线性任务，开发快但流程固定；LangGraph 是 LangChain 团队推出的 Agent 编排引擎，用图（Graph）显式定义节点、边、状态，原生支持循环、分支、持久化和人工介入，适合需要精确控制的多步 Agent。两者是同一个生态的上下层：LangGraph 管流程编排，节点内部可以调用 LangChain 的组件，官方推荐复杂 Agent 用 LangGraph。

> **Q：为什么 LangChain 不适合做复杂 Agent？**
> 因为链（Chain）本质是线性 DAG，流程固定、难以循环，状态管理弱，Agent 跑起来像黑盒，出问题不好调试、不好打断。LangGraph 用图 + 显式状态解决这三个问题：能循环、状态透明、每一步可检查可干预。

> **Q：LangGraph 的核心概念有哪些？**
> State（全局状态）、Node（节点，干活的函数）、Edge（边，连接节点）、Conditional Edge（条件边，决定下一步分支）、Checkpointer（检查点，持久化/断点续跑/Time Travel）。还有一个 END 节点表示流程结束。

> **Q：什么时候用 LangGraph 而不是 LangChain？**
> 当任务需要多步规划、循环调用工具、流程要分支判断、需要持久化或人工审核时，用 LangGraph。简单问答、单次 RAG、固定流程翻译，LangChain 链就够了。

---

## 八、常见追问

**Q1：LangGraph 和 LangChain AgentExecutor 什么关系？**
LangChain 早期的 `AgentExecutor` 把 Agent 循环封装成黑盒，难以控制。LangGraph 是它的升级替代品，官方已将其标记为 legacy，新项目推荐 LangGraph（或 `create_react_agent` 快捷封装）。

**Q2：LangGraph 的状态是怎么共享的？**
定义一个 `TypedDict`（或 Pydantic 模型）作为 State，所有节点读它、改它。节点返回的 dict 会按字段**合并**进全局状态，天然解决多节点数据传递问题。

**Q3：LangGraph 怎么做人工介入（Human-in-the-loop）？**
用 `interrupt_before` / `interrupt_after` 在指定节点前/后暂停图，等待人工确认后通过 checkpointer 恢复执行。适合"AI 生成内容 → 人工审核 → 再提交"的场景。

**Q4：LangGraph 的持久化有什么用？**
配置 Checkpointer 后，图每次执行的状态都会落库。崩溃后可从断点恢复；还能 Time Travel：回退到历史某个状态重新执行，方便调试。

**Q5：RAG 用 LangChain 还是 LangGraph？**
基础 RAG（检索→拼 Prompt→生成）用 LangChain 链足够；但"多轮对话 + 条件路由 + 多文档混合检索"这类复杂 RAG Agent，用 LangGraph 编排更稳。

---

## 相关文件

- [ReAct/知识点.md](./ReAct/知识点.md) — ReAct 范式（LangGraph 常用来实现 ReAct 循环）
- [RAG/知识点.md](./RAG/知识点.md) — RAG 技术
- [Agent完整闭环示例.py](./Agent完整闭环示例.py) — 零依赖手写 Agent 闭环（可对照理解框架做了什么）
