"""
LangGraph 示例拆文件 · 04 · 节点函数
对照自研引擎的 5 种节点类:LLMNode / ToolNode / AgentNode / HumanNode
LangGraph 的节点 = 普通函数:输入整个 State,返回「要更新的字段的 dict」(增量)
"""
import json

from _01_MockLLM import llm
from _02_风控工具 import query_tx, query_device, query_profile
from _03_State定义 import RiskState


# ── ① classify_alert:LLMNode ──────────────────────────────
def classify_alert(state: RiskState) -> dict:
    """用 LLM 判断告警类型"""
    prompt = f"任务:判断告警类型,只输出JSON。告警:{state['alert']}"
    print(f"   [LLM 调用] prompt = {prompt[:50]}...")
    return {"alert_type": json.loads(llm.chat(prompt))["alert_type"]}
    # 注意:返回的是「增量」{"alert_type": ...},LangGraph 负责 merge 进 State;
    # 自研引擎里这一步是 context[current.name] = output,由你的引擎手动存


# ── ③ fetch_data:ToolNode ──────────────────────────────────
def fetch_data(state: RiskState) -> dict:
    """并行调用风控取数工具"""
    u = state["user_id"]
    return {
        "tx": query_tx(u),
        "device": query_device(u),
        "profile": query_profile(u),
    }


# ── ④ analyze:AgentNode ──────────────────────────────────
def analyze(state: RiskState) -> dict:
    """节点内跑 ReAct 小循环"""
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


# ── ⑤ judge:LLMNode ───────────────────────────────────────
def judge(state: RiskState) -> dict:
    """基于证据+数据,让 LLM 生成研判结论"""
    prompt = (f"任务:生成研判结论。证据:{state['evidence']} "
              f"数据:tx={state['tx']} device={state['device']} "
              f"profile={state['profile']}")
    print(f"   [LLM 调用] prompt = {prompt[:50]}...")
    return {"report": json.loads(llm.chat(prompt))}


# ── ⑦ manual_review:HumanNode ─────────────────────────────
def manual_review(state: RiskState) -> dict:
    """人工复核 —— 和自研不同:真挂起由 interrupt_before 实现!"""
    # 图跑到这个节点前自动停,进程可以退出,之后从检查点恢复(见 07-运行流程)
    return {"reviewed_by": "op_77", "decision": "confirmed"}
