# Workflows · Agents · LangChain · LangGraph 四方关系

> 面试高频题。一句话记忆：
> - **Workflows（工作流）= 代码写死的路线**：流程是开发者画好的，LLM 只在指定步骤干活
> - **Agents（智能体）= 模型自己找路**：LLM 在循环里自己决定下一步调什么工具
> - **LangChain / LangGraph = 实现工具**：Workflow 和 Agent 是"设计模式"，这两个框架是"施工工具"
> - 概念出自 Anthropic 2024 年经典文章《Building Effective Agents》

---

## 一、workflows 是干什么的？

**Workflows = 工作流**：用**预先写死的代码路径（predefined code paths）**来编排 LLM 和工具的系统。

翻译成人话：

> 流程是**开发者设计好的**——先做什么、后做什么、什么条件走哪条路，全部写死在代码里。
> LLM 只是流程里的一个"工人"，被安排在哪一步调用就在哪一步调用，**没有决策权**。

```python
# Workflow 的典型形态（伪代码）
def workflow(query):
    intent = classify(query)            # ① 步骤固定
    if intent == "chat":                # ② 分支也是代码写死的
        return chat_llm(query)
    elif intent == "rag":
        docs = retrieve(query)          # ③ 调用检索
        return rag_llm(query, docs)     # ④ 再调模型
```

**特点**：确定性、可预测、便宜、好调试——因为每一步都在开发者掌控中。

---

## 二、Workflow vs Agent（Anthropic 的核心区分）

| 维度 | **Workflows**（工作流） | **Agents**（智能体） |
|---|---|---|
| 谁来定流程 | **开发者**（代码写死路线） | **LLM 自己**（动态决策下一步） |
| LLM 控制权 | 弱（被安排干活） | 强（主导整个执行） |
| 路径 | 预设、确定 | 运行时生成、不可预测 |
| 可靠性 | 高（可控） | 低（可能绕远/出错） |
| 成本/延迟 | 低 | 高（多轮循环、多步调用） |
| 可调试 | 容易 | 难 |
| 适用 | 流程明确、可预测的任务 | 流程不固定、需要临场决策的任务 |
| 例子 | 客服分流、固定格式报告、RAG 问答 | 自主写代码、多工具调研、自主规划任务 |

> **Anthropic 的核心建议（面试加分点）**：
> "能用最简单的方案解决，就别上 Agent。"
> 多轮自主往往带来**更高延迟和账单**，只为了换任务表现。先想清楚值不值，**从 Workflow 开始，按需增加复杂度**。

---

## 三、Workflows 的 5 种经典模式（积木）

Anthropic 把 workflow 拆成 5 种可组合的"积木"：

| 模式 | 干什么 | 例子 |
|---|---|---|
| **1. Prompt Chaining**（提示词链） | 任务拆成步骤，上一步输出喂给下一步 | 先写大纲 → 再写全文 → 再校对 |
| **2. Routing**（路由） | 按输入分类，走不同分支 | 问题分类：闲聊 → 通用模型；技术 → 技术模型 |
| **3. Parallelization**（并行化） | 多个任务同时跑，最后合并 | 分节翻译 + 投票表决 |
| **4. Orchestrator-Workers**（编排者-工人） | 中心 LLM 拆解任务，多个 worker 执行，再汇总 | 拆解"写一份市场报告"给多个 worker 分头写 |
| **5. Evaluator-Optimizer**（评估-优化） | 一个生成、一个挑毛病，循环改进 | AI 写代码 → AI 评审 → 改 → 再评审 |

```
复杂度递进：
  Chain(直线) → Routing(分支) → Parallel(并行)
  → Orchestrator(中心调度) → Evaluator(闭环迭代)
```

---

## 四、和 LangChain / LangGraph 的关系（重点）

**Workflow / Agent 是"设计模式"（概念层），LangChain / LangGraph 是"实现框架"（工具层）。**

```
概念层（干什么）          工具层（怎么实现）
┌────────────────┐      ┌────────────────────────────┐
│  Workflows      │ ───► │  LangChain（链）           │
│  代码写死路线    │      │  Prompt Chaining/Routing  │
└────────────────┘      └────────────────────────────┘
┌────────────────┐      ┌────────────────────────────┐
│  Agents         │ ───► │  LangGraph（图）           │
│  模型自主决策    │      │  条件边+循环+状态+检查点     │
└────────────────┘      └────────────────────────────┘
```

**关键理解**：

| 工具 | 和 Workflow / Agent 的关系 |
|---|---|
| **LangChain** | 链式抽象天然对应 **Workflow**（尤其 Prompt Chaining、Routing）。流程写死在链里 |
| **LangGraph** | 两种都能做：确定性图 = **Workflow**；LLM 决策的条件边循环 = **Agent** |
| **LangGraph Agent** | 正是把 Anthropic 的 "Agent 模式" 落地：`tool → model` 循环，由 LLM 决定何时结束 |

### 对应关系速查

| 概念模式 | 在 LangChain 里 | 在 LangGraph 里 |
|---|---|---|
| Prompt Chaining | `chain = a \| b \| c` | 一条直线 `a→b→c` 的边 |
| Routing | `RunnableBranch` 条件分支 | `add_conditional_edges` 条件边 |
| Orchestrator-Workers | 手写复杂 | 中心节点 + 多个 worker 节点 |
| Evaluator-Optimizer | 难实现 | 两个节点互相指（循环） |
| Agent（自主循环） | 旧 `AgentExecutor`（黑盒，已 legacy） | `tool→model` 循环 + 条件边结束 ✅ |

> 结论：**LangGraph 比 LangChain 表达力更强**——LangChain 主要实现 Workflow，LangGraph 既能实现 Workflow 也能实现 Agent。这也是官方推荐复杂系统用 LangGraph 的原因。

---

## 五、四方对比总表（面试背这张）

| 维度 | Workflows | Agents | LangChain | LangGraph |
|---|---|---|---|---|
| **本质** | 设计模式 | 设计模式 | 实现框架 | 实现框架 |
| **流程谁定** | 开发者写死 | LLM 自主 | 开发者写死（链） | 可写死，可自主（图） |
| **有无循环** | 无（最多 1 次迭代） | 有（自主循环） | 无（线性） | 有（条件边循环） |
| **状态管理** | 靠代码传递 | 靠 Agent 维护 | 弱 | 强（显式 State） |
| **成本** | 低 | 高 | 低 | 中（取决于流程复杂度） |
| **可靠性** | 高 | 中低 | 高 | 中高 |
| **一句话** | 预制路线 | 自主寻路 | 搭链工具 | 搭图工具（全能） |

**记忆口诀**：
> Workflow 是"地图"，Agent 是"自由行"；
> LangChain 是"旅行社一日游"（Workflow），LangGraph 是"DIY 自由行工具"（Workflow + Agent 都能做）。

---

## 六、选型建议（面试按这个答）

| 情况 | 选谁 |
|---|---|
| 流程明确、可预测（问答、翻译、固定报告） | **Workflow**（用 LangChain 或 LangGraph 固定图） |
| 流程会变、要临场决策（多工具调研、写代码） | **Agent**（用 LangGraph 循环图） |
| 想快速出 Demo | LangChain |
| 生产级、要可控可调试可持久化 | LangGraph（Workflow 或 Agent 都行） |
| 不确定 | **先从 Workflow 开始，按需加复杂度**（Anthropic 建议） |

---

## 七、面试速答（背这段）

> **Q：workflows 是干什么的？**
> Workflows（工作流）是一种 LLM 应用设计模式：用预先写死的代码路径编排 LLM 和工具，流程由开发者设计，LLM 只在指定步骤被调用、没有决策权。特点是确定性高、成本低、好调试，适合流程明确的任务。出自 Anthropic《Building Effective Agents》。

> **Q：Workflow 和 Agent 的区别？**
> Workflow 的流程由开发者代码写死，LLM 控制权弱；Agent 的流程由 LLM 在循环里动态决策，自己决定下一步调什么工具。Agent 更灵活但延迟高、成本高、不可预测。Anthropic 建议：能用简单方案就别上 Agent，从 Workflow 开始按需升级。

> **Q：Workflow 和 LangChain / LangGraph 什么关系？**
> Workflow/Agent 是概念（设计模式），LangChain/LangGraph 是实现工具。LangChain 的链式抽象天然实现 Workflow（Prompt Chaining、Routing）；LangGraph 是通用编排引擎，确定性图实现 Workflow，条件边循环实现 Agent。两者是"设计图"和"施工工具"的关系。

> **Q：LangGraph 怎么实现 Agent 模式？**
> 用 StateGraph 定义全局状态，add_node 加"模型节点"和"工具节点"，add_conditional_edges 加条件边——模型节点根据结果决定"继续调工具"还是"结束"，工具节点执行完回到模型节点，形成 `model→tool→model` 循环，直到 LLM 决定结束。

> **Q：Workflows 有哪些经典模式？**
> 五种：Prompt Chaining（链式）、Routing（路由）、Parallelization（并行）、Orchestrator-Workers（编排者-工人）、Evaluator-Optimizer（评估-优化）。可以像积木一样组合。

---

## 相关文件

- [LangChain-LangGraph区别.md](./LangChain-LangGraph区别.md) — 两个框架的对比
- [ReAct/知识点.md](./ReAct/知识点.md) — Agent 循环的具体范式（LangGraph 常用来实现）
- [Agent完整闭环示例.py](./Agent完整闭环示例.py) — 零依赖手写 Agent 闭环（可对照理解）
