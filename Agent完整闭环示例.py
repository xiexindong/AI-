# -*- coding: utf-8 -*-
"""
Agent 完整闭环示例:LLM + Prompt + RAG + ReAct
==================================================

零依赖纯 Python 可运行。场景:公司智能助手,回答"食堂几点关门?"这类真实问题。

四个组件在这个示例里各管什么:
  [LLM]      MiniLLM —— 模拟大模型:理解问题 / 决定下一步 / 生成回答
  [Prompt]   build_prompt —— 系统指令 + 已执行步骤(上下文) 拼装成发给模型的指令
  [RAG]      search_docs —— 检索文档库(模拟向量检索),给模型"查资料"的能力
  [ReAct]    主循环 agent —— 思考→行动→观察→再思考→…直到给出最终回答

完整闭环(以"食堂几点关门"为例):
  用户问题
    → LLM 思考:先查资料(搜索文档库)
    → Action: search_docs("食堂营业时间")        ← RAG
    → Observation: 文档库里没有食堂信息
    → 结果拼回 Prompt,LLM 再思考:查文档没用,调实时接口
    → Action: call_api("canteen_hours")          ← ReAct 的工具调用
    → Observation: {"today":"2026-08-23","hours":"07:30-20:30"}
    → 结果拼回 Prompt,LLM 基于真实数据生成回答
    → 最终答案

真实工程中,把 MiniLLM.chat 换成 GPT/DeepSeek 等模型的 API 调用即可。
"""

# ═══════════════════════════════════════════
# 0. 数据层:文档库(RAG 的数据源,模拟向量数据库)
# ═══════════════════════════════════════════
DOCS = [
    {"id": 1, "tag": "请假",
     "keywords": ["请假", "休假", "年假", "病假"],
     "text": "员工请假流程:提前 3 天在 OA 系统提交申请,经部门主管审批后生效。年假全年 10 天。"},
    {"id": 2, "tag": "报销",
     "keywords": ["报销", "发票", "打款", "费用"],
     "text": "费用报销:因公消费需保留发票,月底前在报销系统上传电子发票,财务 10 个工作日内打款。"},
    {"id": 3, "tag": "工资",
     "keywords": ["工资", "发工资", "薪水", "薪资", "薪酬"],
     "text": "工资发放:每月 15 日发放上月工资,逢节假日顺延至下一工作日。"},
    {"id": 4, "tag": "会议",
     "keywords": ["会议", "会议室", "预订", "订会议室"],
     "text": "会议室预订:通过内部预订系统选择时间,时长超过 2 小时需抄送部门负责人。"},
]


# ═══════════════════════════════════════════
# 1. RAG 检索:search_docs(简化版:关键词打分,模拟向量语义检索)
# ═══════════════════════════════════════════
def search_docs(query: str, top_k: int = 2) -> list[dict]:
    """RAG 的检索环节:问题 → 最相关的文档片段。
    真实工程:query → Embedding → 向量数据库(Chroma/FAISS) → TopK。
    这里用关键词命中数打分模拟"语义相似度":
    - 命中文档关键词 → 相关;一个都不命中 → 返回空(触发 Agent 换工具)。
    """
    scored = []  # 结果列表,元素是元组 (分数, 文档),例如 [(4, {...}), (2, {...})]
    for doc in DOCS:  # 遍历 4 个文档,每个文档算一个"相关度分数"
        # sum(生成器):对文档的每个关键词 kw 逐个判断——
        #   若 kw 出现在问题 query 里,产生一个 2;否则不产生。最后把所有 2 加起来。
        #   例:query="报销流程是什么?",doc 关键词=["报销","发票","打款","费用"]
        #   → "报销"在 query 里(得2) + 其他3个不在 → sum = 2
        #   等价写法:
        #     score = 0
        #     for kw in doc["keywords"]:
        #         if kw in query:
        #             score += 2
        score = sum(2 for kw in doc["keywords"] if kw in query)
        if doc["tag"] in query:  # 额外加分:如果文档的 tag("报销")直接出现在问题里,再加 1 分
            score += 1
        if score > 0:  # 一个关键词都没命中 → 该文档完全不相关,跳过不收录
            scored.append((score, doc))  # 元组 (分数, 文档) 追加进列表,后面靠分数排序
    # sort 排序,key 是"取什么来比大小":
    #   lambda x: -x[0] 的意思是"x 是元组,取第 0 项(分数)并取负",
    #   取负是为了**从大到小**排(默认 sort 是升序,分数取负后升序 = 原分数降序)。
    #   等价写法:
    #     scored.sort(key=lambda x: x[0], reverse=True)
    scored.sort(key=lambda x: -x[0])
    # 列表推导式 [d for _, d in scored[:top_k]]:
    #   scored[:top_k] 取分数最高的前 2 个元组;对每个元组用 _ 和 d 解包:
    #   _ 接住分数(用不到,所以叫 _ 是约定俗成的"忽略"),d 接住文档本身,
    #   最终返回一个"只含文档 dict、按分数从高到低"的列表。
    return [d for _, d in scored[:top_k]]


# ═══════════════════════════════════════════
# 2. 工具层:ReAct 可调用的外部 API(实时数据,文档库里没有)
# ═══════════════════════════════════════════
def call_api(api_name: str, params: dict | None = None):
    """模拟外部系统接口,返回实时数据。
    真实工程:requests.get("https://oa.company.com/api/canteen/hours")
    """
    params = params or {}
    if api_name == "canteen_hours":
        return {"today": "2026-08-23", "hours": "07:30-20:30",
                "note": "周末与节假日 08:00-20:00"}
    if api_name == "weather":
        return {"city": params.get("city", "深圳"), "weather": "多云",
                "temp": "28-34℃", "rain": "无降雨"}
    return None


# ═══════════════════════════════════════════
# 3. Prompt 层:系统指令(约束) + 已执行步骤(上下文)
# ═══════════════════════════════════════════
SYSTEM_PROMPT = (
    "你是\"XX公司行政助手\"。规则:\n"
    "1. 只能依据检索到的文档或工具返回的真实数据回答;\n"
    "2. 查不到的信息必须回答\"暂未查询到\",严禁编造;\n"
    "3. 回答简洁,先给结论再给依据。"
)


def build_prompt(user_query: str, history: list[dict]) -> str:
    """把 系统指令 + 用户问题 + 已经执行的步骤 拼装成发给 LLM 的完整 Prompt。
    这是整个 Agent 里最关键的组装逻辑:RAG 检索结果、ReAct 的观察结果,
    全部通过这里"翻译成文字"喂给模型。
    """
    prompt = f"{SYSTEM_PROMPT}\n\n【用户问题】{user_query}\n\n"
    if history:
        prompt += "【你已执行的步骤】\n"
        for h in history:
            prompt += f"  · {h['type']}: {h['content']}\n"
    prompt += "\n【请决定下一步:要么调用工具,要么给出最终回答】"
    return prompt


# ═══════════════════════════════════════════
# 4. LLM 层:MiniLLM(模拟大模型)
# ═══════════════════════════════════════════
# 真实替换为:
#   resp = openai.ChatCompletion.create(
#       model="gpt-4",
#       messages=[{"role": "system", "content": SYSTEM_PROMPT}, ...])
class MiniLLM:
    """模拟大模型:输入 Prompt,输出"工具调用决策"或"最终回答"。
    这里用规则模拟 LLM 的推理,便于不依赖任何库就能跑通闭环。
    """

    def __init__(self):
        # 模拟"训练知识":只有这一点点,超出范围的都会"编"(幻觉)
        self.knowledge = {
            "入职体检": "新员工报到后一周内到指定三甲医院体检,费用公司承担。",
        }

    # ---- 主入口:输入 prompt + 历史,返回决策 ----
    def chat(self, user_query: str, history: list[dict]) -> dict:
        # 规则 1:已有工具返回了实时数据 → 基于数据生成最终回答
        if history and history[-1]["type"] == "tool_result":
            return {"type": "answer", "content": self._answer_from_tool(history[-1])}

        # 规则 2:已有检索结果(非空) → 基于文档回答
        if history and history[-1]["type"] == "search_result" and history[-1]["content"]:
            return {"type": "answer", "content": self._answer_from_docs(history[-1])}

        # 规则 3:还没搜过 → 先查文档库(RAG)
        if not any(h["type"] == "search_result" for h in history):
            return {"type": "search_docs", "content": user_query}

        # 规则 4:搜过了但没命中 → 调实时接口(ReAct)
        if "食堂" in user_query or "餐" in user_query:
            return {"type": "call_api", "api": "canteen_hours"}
        if "天气" in user_query or "下雨" in user_query or "温度" in user_query:
            return {"type": "call_api", "api": "weather",
                    "params": {"city": "深圳"}}
        # 规则 5:文档没有、工具也没有 → 遵守系统指令,如实说查不到
        return {"type": "answer", "content": "暂未查询到相关信息,建议联系行政部或稍后再试。"}

    # ---- 基于工具数据生成回答 ----
    def _answer_from_tool(self, h: dict) -> str:
        data = h["content"]
        if data.get("hours"):
            return (f"食堂今天({data['today']})营业时间为 {data['hours']}"
                    f"(数据来源:OA 系统实时接口;{data['note']})。")
        if data.get("weather"):
            return (f"{data['city']}今天天气:{data['weather']},"
                    f"气温 {data['temp']},{data['rain']}(数据来源:天气接口)。")
        return "暂未查询到相关信息。"

    # ---- 基于检索文档生成回答 ----
    def _answer_from_docs(self, h: dict) -> str:
        texts = "；".join(d["text"] for d in h["content"])
        return f"根据公司文档:{texts}"

    # ---- 对比用:没有系统指令约束时的"裸模型"(会幻觉) ----
    def bare_chat(self, user_query: str) -> str:
        for kw, ans in self.knowledge.items():
            if kw in user_query:
                return ans
        # 知识里没有的,就"顺着编"(模拟真实 LLM 的幻觉)
        if "食堂" in user_query:
            return "食堂一般是下午 6 点关门,过时不候。"
        if "天气" in user_query:
            return "明天会下大雨,记得带伞。"
        return "这个我不太清楚,但我猜应该是这样的……"


# ═══════════════════════════════════════════
# 5. ReAct 主循环:agent
# ═══════════════════════════════════════════
MAX_STEPS = 5


def run_agent(user_query: str) -> str:
    """完整闭环入口。真实 ReAct 就是这样:while 循环,直到模型输出 answer。"""
    llm = MiniLLM()
    history: list[dict] = []   # 记录每一步(相当于 ReAct 的 scratchpad)

    print(f"\n{'='*68}")
    print(f"用户问题: {user_query}")
    print(f"{'='*68}")

    for step in range(1, MAX_STEPS + 1):
        # ── ① Prompt:把系统指令 + 已执行步骤拼给 LLM ──
        prompt = build_prompt(user_query, history)
        print(f"\n[第{step}轮] 发给 LLM 的 Prompt:")
        print("-" * 68)
        print(prompt)
        print("-" * 68)

        # ── ② LLM 思考:决定下一步 ──
        decision = llm.chat(user_query, history)

        # ── ③ 执行 Action,得到 Observation ──
        if decision["type"] == "search_docs":           # RAG 检索
            hits = search_docs(decision["content"])
            obs = "未找到相关文档" if not hits else \
                  [{"tag": d["tag"], "text": d["text"]} for d in hits]
            print(f"  ▶ Action: search_docs(\"{decision['content']}\")")
            print(f"  ▶ Observation: {obs}")
            history.append({"type": "search_result", "content": hits})

        elif decision["type"] == "call_api":            # ReAct 调工具
            data = call_api(decision["api"], decision.get("params"))
            print(f"  ▶ Action: call_api(\"{decision['api']}\")")
            print(f"  ▶ Observation: {data}")
            history.append({"type": "tool_result", "content": data})

        elif decision["type"] == "answer":              # 生成最终回答
            print(f"  ▶ Action: finish")
            print(f"{'='*68}")
            print(f"最终回答: {decision['content']}")
            print(f"{'='*68}")
            return decision["content"]

    print(f"[达到最大步数 {MAX_STEPS},循环终止]")
    return "暂未查询到相关信息。"


# ═══════════════════════════════════════════
# 6. 对比演示:裸 LLM vs 完整闭环
# ═══════════════════════════════════════════
def demo_compare(user_query: str):
    llm = MiniLLM()
    print(f"\n{'#'*68}")
    print(f"对比:同一个问题,两种做法")
    print(f"问题: {user_query}")
    print(f"{'#'*68}")
    print(f"\n[A] 裸 LLM(只有大脑,没有 RAG / ReAct / 系统指令约束):")
    print(f"    回答: {llm.bare_chat(user_query)}   ← 注意:这是编的(幻觉)")
    print(f"\n[B] LLM + Prompt + RAG + ReAct 完整闭环:")
    run_agent(user_query)


# ═══════════════════════════════════════════
# 7. 主流程
# ═══════════════════════════════════════════
if __name__ == "__main__":
    # 场景 1:文档库没有、需要调实时 API → 完整展示 ReAct 多轮循环
    demo_compare("食堂几点关门?")

    # 场景 2:文档库里有 → 展示 RAG 检索直接回答(一轮搞定)
    print("\n\n" + "●" * 30 + " 第二个问题 " + "●" * 30)
    run_agent("报销流程是什么?")

    # 场景 3:什么都查不到 → 展示系统指令约束(防幻觉)
    print("\n\n" + "●" * 30 + " 第三个问题 " + "●" * 30)
    run_agent("公司下午茶福利是什么?")
