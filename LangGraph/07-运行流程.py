"""
LangGraph 示例拆文件 · 07 · 运行流程
两阶段:① stream 流式跑到人工节点前暂停 → ② 模拟审核员 → 恢复跑完
"""
import json

from _06_建图 import build_graph


if __name__ == "__main__":
    graph = build_graph()
    config = {"configurable": {"thread_id": "case-U1001"}}
    # thread_id = 这次运行的身份;检查点按它存取(对照自研例 2 的 run_id)

    input_state = {
        "alert": "用户 U1001 单笔交易 50 万,命中规则 R12",
        "user_id": "U1001",
        "evidence": [],                     # reducer 字段给个初始空列表,防首写合并报错
    }

    # ════════════════════════════════════════════════════════
    # 阶段 1:stream 逐节点执行,直到人工节点前自动暂停
    # ════════════════════════════════════════════════════════
    print("── 阶段 1:流式执行,直到人工节点前自动暂停 ──")
    for chunk in graph.stream(input_state, config, stream_mode="updates"):
        # chunk 形如 {"classify_alert": {"alert_type": "transaction"}}
        for node_name, delta in chunk.items():
            print(f"   [落库] {node_name} → {json.dumps(delta, ensure_ascii=False)[:60]}")

    # ── 查看暂停在哪(对照自研断点状态的 query 版) ──
    snapshot = graph.get_state(config)
    print(f"\n[暂停] 停在 {snapshot.next} 之前")

    # ════════════════════════════════════════════════════════
    # 阶段 2:审核员确认 → 写入人工结论 → 从断点恢复
    # ════════════════════════════════════════════════════════
    print("\n── 阶段 2:审核员确认,从断点恢复 ──")
    graph.update_state(config, {"reviewed_by": "op_77", "decision": "confirmed"})
    graph.invoke(None, config)              # 传 None = 从上次中断点继续(断点续跑)

    # ════════════════════════════════════════════════════════
    # 最终结果
    # ════════════════════════════════════════════════════════
    final = graph.get_state(config).values
    print("\n════ 最终研判报告(state.report) ════")
    print(json.dumps(final["report"], ensure_ascii=False, indent=2))
    print("人工复核:", {k: final[k] for k in ("reviewed_by", "decision")})
