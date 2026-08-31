# LangGraph 知识点详解 + 风控研判完整示例

> 一个可直接 Python 运行的、零 LLM 依赖的 LangGraph 实战案例。
> 和「简历/Workflow执行引擎示例.py」是同一个风控场景，对照学习效果最佳。

---

## 一、LangGraph 是什么

**LangGraph = 给 LLM Agent 画流程图的库**。

LangChain 是"单次 LLM 调用的链式封装"（LCEL），LangGraph 是"**多轮、带环、带条件分支、可持久化**的 Agent 工作流引擎"。核心能力：

| 能力 | 说明 | 面试关键词 |
|------|------|----------|
| **State** | 强类型的共享"背包" | TypedDict + reducer |
| **Node** | 图上的节点 = 普通 Python 函数 | 输入 State，返回增量 dict |
| **Edge** | 连接节点的线 | 普通边 / 条件边 |
| **Checkpointer** | 状态存档器 | 断点续跑 / HITL |
| **Interrupt** | 真挂起，进程可退出 | 人机协作 |

```
LangChain   → LCEL 链: A → B → C (线性, 单轮)
LangGraph   → StateGraph: 带环、带分支、可持久化的工作流图
自研引擎    → 也是 StateGraph 思想的一种实现
```

---

## 二、和自研引擎 1:1 对照表（核心！）

| 自研引擎（Workflow执行引擎示例.py） | LangGraph（本示例） | 章节 |
|------|------|------|
| context 背包（dict，节点按名字取） | **State(TypedDict)**，节点返回 dict 增量自动 merge | 三 |
| nodes 列表（列表顺序 = 执行顺序） | **add_node + add_edge** 手工连边（无隐式顺序） | 四 |
| CondNode + next_node 分支跳转 | **add_conditional_edges** + 路由函数 | 五 |
| 每轮 save() 打印「落库」 | **graph.stream(updates)** 框架自带逐节点输出 | 七 |
| HumanNode 注释"生产:挂起等人工" | **checkpointer + interrupt_before** 真挂起 | 六 |
| 注释"重启从断点恢复" | **graph.invoke(None, config)** 从中断点继续跑 | 七 |

> **结论：学会 LangGraph，自研工作流引擎的设计思路就通了；反过来也一样。**

---

## 三、State：LangGraph 的"背包"

### 3.1 自研 context vs LangGraph State

```python
# 自研引擎:
context = {}                               # 普通 dict,想塞什么塞什么
context['node_A'] = {'result': 42}          # 整体覆盖
context['evidence'].append('新证据')        # 自己管合并

# LangGraph:
class RiskState(TypedDict):                # 强类型,字段必须先声明
    alert: str
    user_id: str
    evidence: Annotated[list[str], operator.add]   # 🔥 指定合并策略
```

### 3.2 reducer 的威力

普通字段默认是**覆盖**：节点返回 `{"tx": "新流水"}` → 原来的 `tx` 值被替换。

但加了 `Annotated[list[str], operator.add]` 的字段：节点返回 `{"evidence": ["证据A"]}` → LangGraph 自动 `old + new = [旧列表, 新列表]`。

**完整 State 定义** → [03-State定义.py](./LangGraph/03-State定义.py)

### 3.3 坑：schema 过滤

节点返回了 State 里**没声明**的键 → **静默丢弃**。不像自研 context 什么都能塞，字段必须先在 State 里登记。

---

## 四、Node：普通 Python 函数

LangGraph 没有"节点类"，节点就是函数：

```python
def any_node(state: RiskState) -> dict:
    # 输入:整个 State(只读)
    # 返回:要更新的字段的 dict(增量,LangGraph 负责 merge)
    return {"alert_type": "transaction"}    # ← 只返回增量!
```

### 4.1 5 种节点类型映射

| 自研节点 | LangGraph 函数 | 职责 | 代码 |
|---------|---------------|------|------|
| LLMNode | `classify_alert` | 调 LLM 做分类 | [04-节点函数.py](./LangGraph/04-节点函数.py) |
| ToolNode | `fetch_data` | 并行调风控工具取数 | 同上 |
| AgentNode | `analyze` | 节点内跑 ReAct 小循环 | 同上 |
| LLMNode | `judge` | 调 LLM 生成研判报告 | 同上 |
| HumanNode | `manual_review` | 人工复核(真挂起由 interrupt 实现) | 同上 |

---

## 五、Edge：连边 + 路由

### 5.1 普通边 vs 条件边

```python
# 普通边:固定顺序,像自研引擎的"列表下标推进"
builder.add_edge("fetch_data", "analyze")

# 条件边:函数返回值决定去哪,像自研引擎的 CondNode
builder.add_conditional_edges(
    "classify_alert",     # 从哪个节点出来后判断
    route_alert,          # 路由函数
)
```

### 5.2 路由函数

```python
def route_alert(state: RiskState) -> str:
    """返回值 = 节点名;END = 流程结束"""
    if state["alert_type"] == "transaction":
        return "fetch_data"          # 跳到取数节点
    return END                        # 直接结束
```

完整路由代码 → [05-路由函数.py](./LangGraph/05-路由函数.py)

---

## 六、Checkpointer + Interrupt：断点续跑 + 人机协作（面试高频）

### 6.1 自研为什么做不到真挂起？

```python
# 自研 Engine 的 HumanNode 实现:
def run(self, context):
    # time.sleep(3600)  ← 假挂起!进程一直占内存
    # 真要挂起 → 得自己实现状态序列化 + 存 DB + 重启恢复
```

### 6.2 LangGraph 一行搞定

```python
return builder.compile(
    checkpointer=InMemorySaver(),          # 状态存档器(也有 SqliteSaver/PgsqlSaver)
    interrupt_before=["manual_review"],    # 🔥 跑到这个节点之前自动暂停
)
```

### 6.3 运行时两阶段

```
阶段 1:graph.stream(...) 流式执行 → 跑到 manual_review 之前自动停
阶段 2:审核员确认 → graph.update_state(...) 写入人工结论
阶段 3:graph.invoke(None, config) ← 传 None = 从断点继续跑!
```

> **对比自研**：自研 save() 是手动打印，LangGraph checkpointer 是框架自动存档；自研断点恢复是手动查 context，LangGraph 用 thread_id 直接 `invoke(None, config)` 续跑。

完整运行代码 → [07-运行流程.py](./LangGraph/07-运行流程.py)

---

## 七、完整流程图（风控研判案例）

```
                          START
                            │
                            ▼
                  ┌─ classify_alert ─┐
                  │  ① LLM 分类告警   │
                  └─────────┬────────┘
                            │
                    route_alert ②条件边
                   /            \
          transaction            其他类型 → END
                 │
                 ▼
          fetch_data ③取数
          (tx/device/profile)
                 │
                 ▼
          analyze ④ReAct 查证
          (自主选工具 → 循环)
                 │
                 ▼
             judge ⑤LLM 研判
          (生成 risk report)
                 │
             need_human ⑥条件边
            /              \
    conf < 0.9          conf ≥ 0.9
          │                    │
          ▼                    ▼
   manual_review⑦         END
   (interrupt 真挂起)
          │
     审核员确认 → 恢复
          │
          ▼
         END
```

---

## 八、完整建图代码

```python
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver

from _03_State定义 import RiskState
from _04_节点函数 import (classify_alert, fetch_data, analyze, judge, manual_review)
from _05_路由函数 import route_alert, need_human


def build_graph():
    builder = StateGraph(RiskState)          # 声明背包长什么样

    # 注册节点
    builder.add_node("classify_alert", classify_alert)
    builder.add_node("fetch_data", fetch_data)
    builder.add_node("analyze", analyze)
    builder.add_node("judge", judge)
    builder.add_node("manual_review", manual_review)

    # 连边
    builder.add_edge(START, "classify_alert")
    builder.add_conditional_edges("classify_alert", route_alert)
    builder.add_edge("fetch_data", "analyze")
    builder.add_edge("analyze", "judge")
    builder.add_conditional_edges("judge", need_human)

    # 编译 + checkpointer + interrupt
    return builder.compile(
        checkpointer=InMemorySaver(),
        interrupt_before=["manual_review"],
    )
```

完整文件 → [06-建图.py](./LangGraph/06-建图.py)

---

## 九、运行输出样例

```
── 阶段 1:流式执行,直到人工节点前自动暂停 ──
   [LLM 调用] prompt = 任务:判断告警类型,只输出JSON。告警:用户 U1001 单笔交易...
   [落库] classify_alert → {"alert_type": "transaction"}
   [落库] fetch_data → {"tx": "12 笔共 63 万...", "device": "24h 内登录设备 2→5 台...", "profile": "..."}
   [ReAct 1] Thought: 先查交易流水 → Action: query_tx
   [ReAct 1] Observation: 12 笔共 63 万,其中 3 笔深夜大额
   [ReAct 2] Thought: 流水异常,再查设备 → Action: query_device
   [ReAct 2] Observation: 24h 内登录设备 2 台 → 5 台,新增 3 台 rooted
   [ReAct 3] Thought: 证据足够,收敛 → Action: FINISH
   [落库] analyze → {"evidence": ["12 笔共 63 万...", "24h 内登录设备..."]}
   [LLM 调用] prompt = 任务:生成研判结论。证据:['12 笔共 63 万...' ...
   [落库] judge → {"report": {"risk_level": "HIGH", "confidence": 0.87, ...}}

[暂停] 停在 ('manual_review',) 之前

── 阶段 2:审核员确认,从断点恢复 ──

════ 最终研判报告(state.report) ════
{
  "risk_level": "HIGH",
  "confidence": 0.87,
  "conclusions": [...],
  "suggestion": "转人工复核(置信度 < 0.9 强制人工)"
}
人工复核: {'reviewed_by': 'op_77', 'decision': 'confirmed'}
```

---

## 十、文件结构

```
LangGraph/
├── 示例.py                 ← 原始完整文件(单文件可运行)
├── langgraph.md            ← 本知识点文档
├── 01-MockLLM.py           ← MockLLM 类(按关键字返回 JSON)
├── 02-风控工具.py          ← query_tx / query_device / query_profile
├── 03-State定义.py         ← RiskState(TypedDict + reducer)
├── 04-节点函数.py          ← classify / fetch_data / analyze / judge / manual_review
├── 05-路由函数.py          ← route_alert / need_human
├── 06-建图.py              ← build_graph() + compile(checkpointer + interrupt)
└── 07-运行流程.py          ← 两阶段运行:stream → 暂停 → update_state → invoke 续跑
```

---

## 十一、环境 & 运行

```bash
pip install langgraph          # 示例在 1.2.11 下验证通过

# 方式 1:直接跑单文件
python LangGraph/示例.py

# 方式 2:跑拆分文件(注意 import 是 _01_MockLLM 格式,需在 LangGraph/ 目录下执行)
cd LangGraph
python 07-运行流程.py
```

---

## 十二、面试要点

### 12.1 LangGraph vs LangChain

| 维度 | LangChain (LCEL) | LangGraph |
|------|-------------------|-----------|
| 控制流 | 线性链 `A \| B \| C` | 图结构,可环可分支 |
| 多轮 | 单层嵌套 | 图上自然循环 |
| 持久化 | 无原生支持 | checkpointer 原生支持 |
| 人机协作 | 无原生 | interrupt_before 一行搞定 |
| 适用 | 简单 RAG / 单次 agent | 生产级多步骤 Agent |

### 12.2 Checkpointer 三种级别

| 实现 | 持久化 | 场景 |
|------|--------|------|
| `InMemorySaver` | 进程内,重启丢 | 开发/测试 |
| `SqliteSaver` | 单机持久 | 单机生产 |
| `PostgresSaver` | 分布式持久 | 多副本生产 |

### 12.3 HITL (Human In The Loop) 三条路

```
1. interrupt_before = ["节点名"]   ← 跑到某节点前暂停(本示例)
2. interrupt_after = ["节点名"]    ← 某节点跑完后暂停
3. graph.invoke(None, config)      ← 从上次中断点恢复
```

### 12.4 State reducer 的面试追问

**Q:为什么默认是覆盖？**
> 因为覆盖是最直观的语义——节点返回什么,State 就变成什么。只有确实需要"累加"的字段(evidence/logs)才用 reducer。

**Q:reducer 函数签名？**
> 接收两个参数 `(old, new)`，返回合并后的值。`operator.add` 是 `list + list` 的标准合并。

**Q:为什么 LangGraph 要 schema 过滤？**
> 防止节点函数"随手塞个字段"导致 State 越来越乱。强制声明 = 强类型文档 + 运行时保护。

---

## 十三、记忆口诀

```
State 要 TypedDict, reducer 加 Annotated,     ← 强类型背包,合并策略可选
Node 就普通函数,输入 State 返回 增量 dict,      ← 节点零魔法,纯 Python
Edge 手工连, 条件边靠路由函数,                 ← 无隐式顺序,图结构一目了然
Checkpointer 存存档, interrupt 真挂起,          ← 断点续跑 + 人机协作一行搞定
invoke None 续跑, thread_id 当身份证。          ← 多会话隔离靠它
```
