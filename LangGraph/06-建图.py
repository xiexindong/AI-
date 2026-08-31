"""
LangGraph 示例拆文件 · 06 · 建图
对照自研引擎的 build_workflow()
自研:「列表顺序 = 执行顺序」靠引擎下标推进
LangGraph:「没有隐式顺序」,每条边必须显式声明 —— 图结构一目了然
"""
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from _03_State定义 import RiskState
from _04_节点函数 import (
    classify_alert, fetch_data, analyze, judge, manual_review,
)
from _05_路由函数 import route_alert, need_human


def build_graph():
    builder = StateGraph(RiskState)          # 传入 State 类型 = 声明「背包长什么样」

    # ── 注册节点(名字, 函数) ──
    builder.add_node("classify_alert", classify_alert)
    builder.add_node("fetch_data", fetch_data)
    builder.add_node("analyze", analyze)
    builder.add_node("judge", judge)
    builder.add_node("manual_review", manual_review)

    # ── 连边 ──
    builder.add_edge(START, "classify_alert")            # 入口边
    builder.add_conditional_edges(                       # ② 条件边 = CondNode
        "classify_alert",                                #   从哪个节点出来后判断
        route_alert,                                     #   路由函数(纯代码)
    )                                                    #   返回什么名字就走哪条边
    builder.add_edge("fetch_data", "analyze")            # 普通边:固定顺序
    builder.add_edge("analyze", "judge")
    builder.add_conditional_edges("judge", need_human)   # ⑥ 条件边

    # ═══════════════════════════════════════════════════════════════
    # compile() —— 把声明式 builder "封口" 成能跑的可执行图
    #
    # builder 对象只负责收集声明(add_node / add_edge),本身不能执行;
    # compile 之后产出 CompiledGraph,才拥有 .invoke() / .stream() 等运行能力。
    # 类比:自研引擎里手写 nodes 列表 → 引擎把它拓扑排序成可执行图,compile 就是那一步。
    # ═══════════════════════════════════════════════════════════════
    #
    # 第一个参数 checkpointer=InMemorySaver()
    # ────────────────────────────────────────
    # 给图装一个"自动存档外挂"。没它:图跑完 state 就没了,也没法从中间恢复。
    # 有了它:每跑完一个节点,自动把当前 state 快照 + next 节点信息存下来。
    # 存档按 config["configurable"]["thread_id"] 分桶(thread_id = 会话身份证,不会串)。
    # 三种实现:
    #   InMemorySaver   内存存档,进程死就丢 —— 本示例用它,开发测试够用
    #   SqliteSaver     单机磁盘持久 —— 单机生产
    #   PostgresSaver   分布式数据库 —— 多副本生产
    #
    # 第二个参数 interrupt_before=["manual_review"]
    # ──────────────────────────────────────────────
    # HITL(Human In The Loop)人机协作的核心机制:跑到 manual_review 节点【之前】自动暂停。
    # 和自研引擎的假挂起(time.sleep 占内存)不同 —— 这里是真挂起:
    #   ① 图跑到 manual_review 上一个节点(judge)执行完
    #   ② 自动存档(靠 checkpointer)
    #   ③ 自动停!进程可以退出,内存释放
    #   ④ 审核员几分钟后点"确认" → graph.invoke(None, config) 从断点续跑
    # 还有 interrupt_after=["节点名"]:跑完某节点之后才暂停。
    # ═══════════════════════════════════════════════════════════════
    return builder.compile(
        checkpointer=InMemorySaver(),           # 自动存档外挂(断点续跑的基础)
        interrupt_before=["manual_review"],     # 跑到这之前真挂起(HITL)
    )


if __name__ == "__main__":
    graph = build_graph()
    print("建图成功!节点:", graph.nodes)
