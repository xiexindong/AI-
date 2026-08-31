"""
LangGraph 示例拆文件 · 03 · State 定义
对照自研引擎的「context 背包」——但 LangGraph 的 State 是**强类型 + 带 reducer 的 TypedDict**
"""
from typing import Annotated, TypedDict
import operator


class RiskState(TypedDict):
    # ── 输入字段(main 里塞进来的) ──
    alert: str                       # 告警原文:① 分类的原料
    user_id: str                     # 被研判用户:③ 取数、④ 查证的参数

    # ── 中间结果(各节点返回的增量,LangGraph 自动 merge 进来) ──
    alert_type: str                  # ① 分类节点的输出
    tx: str                          # ③ 交易流水
    device: str                      # ③ 设备指纹
    profile: str                     # ③ 用户画像

    # ── evidence 带 reducer:节点返回的列表「追加」而不是「覆盖」 ──
    # 自研引擎:context[node.name] = output 是整体覆盖
    # LangGraph 默认也覆盖,但 Annotated + reducer 可指定合并策略(这是它的特色)
    evidence: Annotated[list[str], operator.add]

    # ── 最终产出 ──
    report: dict                     # ⑤ 研判报告
    reviewed_by: str                 # ⑦ 人工复核人
    decision: str                    # ⑦ 人工结论(确认/驳回)


# 坑:LangGraph 按 State schema 过滤字段
# 节点返回了 schema 里没声明的键 → 静默丢弃(不像自研 context 什么都能塞)
# 字段必须先在 State 里登记

if __name__ == "__main__":
    s: RiskState = {
        "alert": "测试告警",
        "user_id": "U1001",
        "evidence": [],
    }
    print("State 初始:", s)
