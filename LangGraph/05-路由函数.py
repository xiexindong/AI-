"""
LangGraph 示例拆文件 · 05 · 路由函数
对照自研引擎的 CondNode:纯代码判断,返回「下一个节点名」或 END
"""
from langgraph.graph import END

from _03_State定义 import RiskState


# ── ② route_alert ─────────────────────────────────────────
def route_alert(state: RiskState) -> str:
    """告警分类后分流 —— transaction 类型才进研判,其他直接结束"""
    # 返回值 = 要跳去的节点名;返回 END = 流程结束
    # 自研:CondNode.branch() 返回 next_true/next_false,引擎 find() 查表
    if state["alert_type"] == "transaction":
        return "fetch_data"
    return END                                                   # 登录分支本例略


# ── ⑥ need_human ──────────────────────────────────────────
def need_human(state: RiskState) -> str:
    """研判报告置信度 < 0.9 → 转人工,否则自动落库(本例略)"""
    if state["report"]["confidence"] < 0.9:
        return "manual_review"
    return END
