# -*- coding: utf-8 -*-
"""
Function Calling(工具调用) 全流程示例
========================================
零依赖纯 Python 可运行。演示核心机制:

  用户提问 → 模型决策(选函数+生成参数) → 代码执行 → 结果回传 → 模型总结

流程拆解:
  ① 定义工具清单(函数 Schema):告诉模型"有哪些函数可用、参数长什么样"
  ② 模型返回 tool_call:{name: 函数名, arguments: JSON 参数}   ← 模型只出这个
  ③ 你的代码校验参数并【真正执行】函数                         ← 关键:执行在代码侧
  ④ 把执行结果作为 tool 消息回传模型
  ⑤ 模型基于真实结果生成最终回答

真实工程中,把 simulate_llm_decision 换成 GPT/DeepSeek 等模型的
tool calling 接口返回即可,其余逻辑完全一致。
"""

import json

# ═══════════════════════════════════════════
# 1. 工具清单:函数 Schema(给模型看的"说明书")
# ═══════════════════════════════════════════
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某个城市的实时天气(当用户问天气/下雨/温度时用)",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名,如 深圳"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_canteen_hours",
            "description": "查询公司食堂当天营业时间(当用户问食堂/几点关门时用)",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# ═══════════════════════════════════════════
# 2. 真实函数:真正执行的代码(模型永远不会自己跑它)
# ═══════════════════════════════════════════
def get_weather(city: str) -> dict:
    """真实工程:requests.get(f"https://api.weather.com/?city={city}")"""
    return {"city": city, "weather": "多云", "temp": "28-34℃", "rain": "无降雨"}


def get_canteen_hours() -> dict:
    """真实工程:调用 OA 系统接口"""
    return {"today": "2026-08-23", "hours": "07:30-20:30",
            "note": "周末与节假日 08:00-20:00"}


# 工具注册表:name → 真实函数(模型选完名字,代码从这里找函数执行)
TOOL_REGISTRY = {
    "get_weather": get_weather,
    "get_canteen_hours": get_canteen_hours,
}


# ═══════════════════════════════════════════
# 3. 模拟 LLM:根据用户问题返回"决策"
#    (真实工程:调 openai/DeepSeek 等接口,返回 resp.tool_calls)
# ═══════════════════════════════════════════
def simulate_llm_decision(user_query: str) -> dict:
    """模拟模型的 tool calling 返回。
    返回三种结果之一:
      {"type": "tool_call", "name": ..., "arguments": {...}}  要调工具
      {"type": "answer", "content": ...}                      不需要工具,直接答
    """
    if "天气" in user_query or "下雨" in user_query or "温度" in user_query:
        # 模型从 TOOLS 里选中 get_weather,并生成合法参数
        return {"type": "tool_call", "name": "get_weather",
                "arguments": {"city": "深圳"}}
    if "食堂" in user_query or "餐" in user_query:
        return {"type": "tool_call", "name": "get_canteen_hours",
                "arguments": {}}
    # 普通闲聊 → 不需要工具,直接回答
    return {"type": "answer", "content": "你好!我可以查天气、查食堂营业时间。"}


def simulate_llm_summarize(user_query: str, tool_result: dict) -> str:
    """模拟模型拿到工具结果后,基于【真实数据】生成最终回答。"""
    if "weather" in tool_result:
        d = tool_result
        return (f"{d['city']}今天天气:{d['weather']},气温 {d['temp']},"
                f"{d['rain']}(数据来源:天气接口)")
    if "hours" in tool_result:
        d = tool_result
        return (f"食堂今天({d['today']})营业时间 {d['hours']}"
                f"({d['note']})(数据来源:OA 系统)")
    return "已查询,暂未获取到数据。"


# ═══════════════════════════════════════════
# 4. 核心:Function Calling 执行器(面试考的就是这个流程)
# ═══════════════════════════════════════════
def run_function_calling(user_query: str) -> str:
    """完整走一遍: 决策 → 执行 → 回传 → 总结"""
    print(f"\n{'='*68}")
    print(f"用户提问: {user_query}")
    print(f"{'='*68}")

    # ① 模型决策:选函数 + 生成参数
    decision = simulate_llm_decision(user_query)
    if decision["type"] == "answer":
        print(f"[模型] 不需要工具,直接回答:")
        print(f"  回答: {decision['content']}")
        return decision["content"]

    print(f"[模型] 决定调用工具:")
    print(f"  tool_call = {json.dumps(decision, ensure_ascii=False)}")

    # ② 代码校验 + 执行(模型只出意图,执行永远在代码侧!)
    name, args = decision["name"], decision["arguments"]
    print(f"[代码] 校验参数并执行 {name}({args})")
    result = TOOL_REGISTRY[name](**args)
    print(f"[代码] 工具返回: {json.dumps(result, ensure_ascii=False)}")

    # ③ 把结果回传模型(真实工程:追加 role=tool 消息再调一次接口)
    print(f"[代码] 将结果回传给模型...")
    answer = simulate_llm_summarize(user_query, result)

    # ④ 模型基于真实数据总结
    print(f"[模型] 基于工具结果总结:")
    print(f"  回答: {answer}")
    return answer


# ═══════════════════════════════════════════
# 5. 主流程
# ═══════════════════════════════════════════
if __name__ == "__main__":
    # 场景 1:需要调工具(天气)
    run_function_calling("今天深圳天气怎么样?")

    # 场景 2:需要调工具(食堂)
    print("\n" + "●" * 30 + " 第二个问题 " + "●" * 30)
    run_function_calling("食堂几点关门?")

    # 场景 3:不需要工具,直接回答
    print("\n" + "●" * 30 + " 第三个问题 " + "●" * 30)
    run_function_calling("你好,你会做什么?")

    print("\n\n[对比提示] 在 Agent完整闭环示例.py 里,")
    print("call_api('canteen_hours') 就是这个机制;加上 RAG 检索和 ReAct 循环,")
    print("就拼成了完整的 Agent 闭环。")
