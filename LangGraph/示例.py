"""
LangGraph 版风控研判编排 · 与自研引擎 1:1 对照(零 LLM 依赖,直接 python 运行)
================================================================================
和「简历/Workflow执行引擎示例.py」是同一个风控案例、同一套 MockLLM,
用 LangGraph 重写一遍——每个概念都能在自研引擎里找到对应物,学完即迁移。

  自研引擎(Workflow执行引擎示例.py)      →  LangGraph 本文件
  ─────────────────────────────────────────────────────────────────
  context 背包(dict,节点按名字取)        →  State(TypedDict),节点返回 dict 增量更新
  nodes 列表(列表顺序 = 执行顺序)        →  add_node + add_edge 手工连边
  CondNode + next_node 分支跳转          →  add_conditional_edges + 路由函数
  每轮 save() 打印「落库」               →  graph.stream(updates) 自带逐节点输出
  HumanNode 注释「生产:挂起等人工」      →  checkpointer + interrupt_before 真实现了!
  注释「重启从断点恢复」                 →  invoke(None, config) 从中断点继续跑

环境:pip install langgraph   (本机已装 1.2.11)
运行:python 示例.py
"""

import json
import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


# ─────────────────────────────────────────────
# 1. MockLLM(和自研引擎同款:按「任务」关键字返回 JSON)
# ─────────────────────────────────────────────
class MockLLM:
    def chat(self, prompt: str) -> str:
        if "任务:判断告警类型" in prompt:            # ① classify 会发来
            return '{"alert_type": "transaction", "confidence": 0.95}'

        if "任务:自主查证" in prompt:                # ④ analyze 的 ReAct 决策轮
            if "已有 0 条证据" in prompt:
                return '{"thought": "先查交易流水", "action": "query_tx"}'
            if "已有 1 条证据" in prompt:
                return '{"thought": "流水异常,再查设备", "action": "query_device"}'
            return '{"thought": "证据足够,收敛", "action": "FINISH"}'

        if "任务:生成研判结论" in prompt:            # ⑤ judge 会发来
            return json.dumps({
                "risk_level": "HIGH",
                "confidence": 0.87,
                "conclusions": [
                    {"point": "近 7 天累计交易 63 万,3 笔深夜大额",
                     "source": "query_tx 返回"},
                    {"point": "24h 内新增 3 台 rooted 设备登录",
                     "source": "query_device 返回"},
                ],
                "suggestion": "转人工复核(置信度 < 0.9 强制人工)",
            }, ensure_ascii=False)

        raise ValueError("MockLLM 不认识这个 prompt: " + prompt[:40])


llm = MockLLM()


# ─────────────────────────────────────────────
# 2. 风控工具(和自研引擎同款)
# ─────────────────────────────────────────────
def query_tx(user_id):
    return "12 笔共 63 万,其中 3 笔深夜大额"


def query_device(user_id):
    return "24h 内登录设备 2 台 → 5 台,新增 3 台 rooted"


def query_profile(user_id):
    return "历史月均交易 8 万"


# ─────────────────────────────────────────────
# 3. State 定义(对照自研引擎的 context 背包)
# ─────────────────────────────────────────────
class RiskState(TypedDict):
    # 输入字段(main 里塞进来的,对照自研 context 初始值)
    alert: str                       # 告警原文:① 分类的原料
    user_id: str                     # 被研判用户:③ 取数、④ 查证的参数
    # 中间结果(各节点返回的增量,LangGraph 自动 merge 进来)
    alert_type: str                  # ① 分类节点的输出(自研里是整个 dict,这里拆平)
    tx: str                          # ③ 交易流水
    device: str                      # ③ 设备指纹
    profile: str                     # ③ 用户画像
    # evidence 带 reducer:节点返回的列表「追加」而不是「覆盖」——
    # 对照自研引擎:context[node.name] = output 是整体覆盖;
    # LangGraph 默认也覆盖,但 Annotated + reducer 可指定合并策略(这是它的特色)
    evidence: Annotated[list[str], operator.add]
    report: dict                     # ⑤ 研判报告
    reviewed_by: str                 # ⑦ 人工复核人
    decision: str                    # ⑦ 人工结论(确认/驳回)
    # 坑:LangGraph 按 State schema 过滤字段——节点返回了 schema 里没声明的键会被
    # 静默丢弃(不像自研引擎的 context 背包什么都能塞),字段必须先在 State 里登记


# ─────────────────────────────────────────────
# 4. 节点函数(对照自研的 5 种节点类)
#    LangGraph 的节点就是普通函数:输入整个 State,返回「要更新的字段的 dict」
# ─────────────────────────────────────────────
def classify_alert(state: RiskState) -> dict:                    # ① 对照 LLMNode
    prompt = f"任务:判断告警类型,只输出JSON。告警:{state['alert']}"
    print(f"   [LLM 调用] prompt = {prompt[:50]}...")
    return {"alert_type": json.loads(llm.chat(prompt))["alert_type"]}
    # 注意:返回的是「增量」{"alert_type": ...},LangGraph 负责 merge 进 State;
    # 自研引擎里这一步是 context[current.name] = output,由你的引擎手动存


def fetch_data(state: RiskState) -> dict:                        # ③ 对照 ToolNode
    u = state["user_id"]
    return {"tx": query_tx(u), "device": query_device(u), "profile": query_profile(u)}


def analyze(state: RiskState) -> dict:                           # ④ 对照 AgentNode
    # 节点内跑 ReAct 小循环(和自研同构)。LangGraph 也能把 ReAct 画成
    # 「外层循环边」(条件边指回自己),生产更常用那种画法,这里保持 1:1 对照
    obs: list[str] = []
    allowed = {"query_tx": query_tx, "query_device": query_device}
    for i in range(8):                                           # max_iter 防跑飞
        prompt = (f"任务:自主查证。已有 {len(obs)} 条证据:{obs} "
                  f"可选工具:{list(allowed)}")
        decision = json.loads(llm.chat(prompt))
        print(f"   [ReAct {i + 1}] Thought: {decision['thought']}"
              f" → Action: {decision['action']}")
        if decision["action"] == "FINISH":
            break
        obs.append(allowed[decision["action"]](state["user_id"]))
        print(f"   [ReAct {i + 1}] Observation: {obs[-1]}")
    return {"evidence": obs}        # reducer=operator.add → 追加进 State.evidence


def judge(state: RiskState) -> dict:                             # ⑤ 对照 LLMNode
    prompt = (f"任务:生成研判结论。证据:{state['evidence']} "
              f"数据:tx={state['tx']} device={state['device']} "
              f"profile={state['profile']}")
    print(f"   [LLM 调用] prompt = {prompt[:50]}...")
    return {"report": json.loads(llm.chat(prompt))}


def manual_review(state: RiskState) -> dict:                     # ⑦ 对照 HumanNode
    # 与自研不同:这里不用 time.sleep 模拟挂起!
    # 真挂起由 compile(interrupt_before=["manual_review"]) 实现——
    # 图跑到这个节点前自动停,进程可以退出,之后从检查点恢复(见 main 第二阶段)
    return {"reviewed_by": "op_77", "decision": "confirmed"}


# ─────────────────────────────────────────────
# 5. 路由函数(对照自研的 CondNode:纯代码判断,返回「下一个节点名」)
# ─────────────────────────────────────────────
def route_alert(state: RiskState) -> str:                        # ② 对照 route
    # 返回值 = 要跳去的节点名;返回 END = 流程结束
    # 对照自研:CondNode.branch() 返回 next_true/next_false,引擎 find() 查表
    if state["alert_type"] == "transaction":
        return "fetch_data"
    return END                                                   # 登录分支本例略


def need_human(state: RiskState) -> str:                         # ⑥ 对照 need_human
    if state["report"]["confidence"] < 0.9:
        return "manual_review"
    return END                                                   # ≥0.9 自动落库(略)


# ─────────────────────────────────────────────
# 6. 建图(对照自研的 build_workflow():节点列表 + 顺序声明)
#    自研:「列表顺序 = 执行顺序」靠引擎下标推进;
#    LangGraph:「没有隐式顺序」,每条边必须显式声明——图结构一目了然
# ─────────────────────────────────────────────
def build_graph():
    builder = StateGraph(RiskState)          # 传入 State 类型 = 声明「背包长什么样」

    builder.add_node("classify_alert", classify_alert)   # 注册节点(名字, 函数)
    builder.add_node("fetch_data", fetch_data)
    builder.add_node("analyze", analyze)
    builder.add_node("judge", judge)
    builder.add_node("manual_review", manual_review)

    builder.add_edge(START, "classify_alert")            # 入口边(对照:引擎固定从 nodes[0] 开始)
    builder.add_conditional_edges(                       # ② 条件边 = CondNode
        "classify_alert",                                #   从哪个节点出来后判断
        route_alert,                                     #   路由函数(纯代码)
    )                                                    #   返回什么名字就走哪条边
    builder.add_edge("fetch_data", "analyze")            # 普通边:固定顺序(对照线性推进)
    builder.add_edge("analyze", "judge")
    builder.add_conditional_edges("judge", need_human)   # ⑥ 条件边

    # checkpointer = 状态存档器:每个节点跑完自动存档(对照自研 save() 那行的真实现)
    # interrupt_before = 跑到 manual_review 之前自动暂停(HITL 人机协作)
    return builder.compile(
        checkpointer=InMemorySaver(),
        interrupt_before=["manual_review"],
    )


# ─────────────────────────────────────────────
# 7. 运行:第一阶段流式跑到「人工前」暂停 → 第二阶段模拟审核员 → 恢复跑完
# ─────────────────────────────────────────────
if __name__ == "__main__":
    graph = build_graph()
    config = {"configurable": {"thread_id": "case-U1001"}}
    # thread_id = 这次运行的身份;检查点按它存取(对照自研例 2 的 run_id)

    input_state = {
        "alert": "用户 U1001 单笔交易 50 万,命中规则 R12",
        "user_id": "U1001",
        "evidence": [],                     # reducer 字段给个初始空列表,防首写合并报错
    }

    # ── 阶段 1:stream 逐节点执行(对照自研每轮的 [落库] 打印,这是框架自带的)──
    print("── 阶段 1:流式执行,直到人工节点前自动暂停 ──")
    for chunk in graph.stream(input_state, config, stream_mode="updates"):
        # chunk 形如 {"classify_alert": {"alert_type": "transaction"}}
        for node_name, delta in chunk.items():
            print(f"   [落库] {node_name} → {json.dumps(delta, ensure_ascii=False)[:60]}")

    # ── 查看暂停在哪(对照自研例 2 断点状态的「query 版」)──
    snapshot = graph.get_state(config)
    print(f"\n[暂停] 停在 {snapshot.next} 之前,已到的人工决策点 next={snapshot.next}")

    # ── 阶段 2:模拟审核员在后台点「确认」→ 写入人工结论 → 恢复执行 ──
    print("\n── 阶段 2:审核员确认,从断点恢复 ──")
    graph.update_state(config, {"reviewed_by": "op_77", "decision": "confirmed"})
    graph.invoke(None, config)              # 传 None = 从上次中断点继续(断点续跑)

    # ── 最终结果 ──
    final = graph.get_state(config).values
    print("\n════ 最终研判报告(state.report) ════")
    print(json.dumps(final["report"], ensure_ascii=False, indent=2))
    print("人工复核:", {k: final[k] for k in ("reviewed_by", "decision")})
