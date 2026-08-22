"""
ReAct 范式最小可运行实现
========================

用 Python 模拟一个 ReAct 循环:大模型边想边干,通过调用工具解决任务。

【JS 类比】
整体结构等价于 JS 里:
    while (true) {
      const step = await llm(promptWithContext);
      if (step.action === 'finish') return step.answer;
      const obs = await tools[step.action](step.input);
      observations.push(obs);
    }
"""

import re
import json

# ─────────────────────────────────────────────
# 1. 模拟一个 LLM(真实场景换成 OpenAI/通义千问等 SDK 调用)
# ─────────────────────────────────────────────

# 模拟模型针对每一步的"思考 + 行动"输出
# 真实场景下,这是 LLM 根据 prompt + 历史观察 生成的内容
MOCK_LLM_RESPONSES = [
    # 第 1 轮:模型决定先查上海天气
    """Thought: 我需要先查上海今天的天气,才能判断要不要带伞。
Action: search_weather
Action Input: {"city": "上海"}
""",
    # 第 2 轮:看到晴天,得出结论,选择结束
    """Thought: 上海今天 32 度晴天,降水概率只有 5%,不需要带伞。
Action: finish
Action Input: {"answer": "不用带伞,上海今天是晴天"}
""",
]


def llm_respond(prompt: str, step_index: int) -> str:
    """模拟 LLM 调用,按 step 顺序返回预设回复。"""
    if step_index >= len(MOCK_LLM_RESPONSES):
        # 兜底:防止无限循环
        return 'Action: finish\nAction Input: {"answer": "已达到最大轮次"}'
    return MOCK_LLM_RESPONSES[step_index]


# ─────────────────────────────────────────────
# 2. 定义工具集合(真实场景会有搜索、数据库、API 等)
# ─────────────────────────────────────────────


def search_weather(action_input: dict) -> str:
    """模拟天气查询工具"""
    city = action_input.get("city", "未知")
    # 真实场景:调用天气 API
    weather_db = {
        "上海": "上海今天 32℃,晴天,降水概率 5%",
        "北京": "北京今天 28℃,多云,降水概率 20%",
    }
    return weather_db.get(city, f"{city} 暂无天气数据")


# 工具注册表:Action 名 → 函数
# JS 类比:const tools = { search_weather: (input) => ... }
TOOLS = {
    "search_weather": search_weather,
}


# ─────────────────────────────────────────────
# 3. 解析模型输出,抽出 Thought / Action / Action Input
# ─────────────────────────────────────────────


def parse_response(response: str) -> dict:
    """
    把模型输出的多行文本解析成结构化字段。

    输入示例:
        Thought: 我需要查天气
        Action: search_weather
        Action Input: {"city": "上海"}

    输出:
        {
            "thought": "我需要查天气",
            "action": "search_weather",
            "action_input": {"city": "上海"}
        }
    """
    thought = re.search(r"Thought:\s*(.+)", response)
    action = re.search(r"Action:\s*(\w+)", response)
    action_input = re.search(r"Action Input:\s*(\{.*?\})", response, re.DOTALL)

    return {
        "thought": thought.group(1).strip() if thought else "",
        "action": action.group(1).strip() if action else "",
        "action_input": json.loads(action_input.group(1)) if action_input else {},
    }


# ─────────────────────────────────────────────
# 4. ReAct 核心循环
# ─────────────────────────────────────────────


def react_agent(task: str, max_steps: int = 5) -> str:
    """
    ReAct 主循环:思考 → 行动 → 观察,直到 finish。

    JS 类比整体结构:
        let observations = [];
        for (let i = 0; i < max_steps; i++) {
            const step = await llm(buildPrompt(task, observations));
            const parsed = parse(step);
            if (parsed.action === 'finish') return parsed.answer;
            const obs = tools[parsed.action](parsed.action_input);
            observations.push(obs);
        }
    """
    print(f"🎯 任务: {task}\n" + "=" * 60)

    observations = []  # 收集每轮 Observation,作为下一轮 prompt 上下文

    for step in range(max_steps):
        # ① 拼 prompt:任务 + 历史观察
        prompt = f"任务: {task}\n已知信息: {observations}"
        # ② 调 LLM 拿到 Thought + Action
        response = llm_respond(prompt, step)
        parsed = parse_response(response)

        print(f"\n【第 {step + 1} 轮】")
        print(f"Thought: {parsed['thought']}")

        # ③ 判断是否结束
        if parsed["action"] == "finish":
            answer = parsed["action_input"].get("answer", "")
            print(f"Action: finish ✅")
            print(f"\n{'=' * 60}\n🏁 最终答案: {answer}")
            return answer

        # ④ 调用工具,拿到 Observation
        tool_fn = TOOLS.get(parsed["action"])
        if tool_fn is None:
            observation = f"工具 {parsed['action']} 不存在"
        else:
            observation = tool_fn(parsed["action_input"])

        print(f"Action: {parsed['action']}({parsed['action_input']})")
        print(f"Observation: {observation}")

        observations.append(observation)

    return "达到最大轮次,未得出结论"


# ─────────────────────────────────────────────
# 5. 运行
# ─────────────────────────────────────────────

if __name__ == "__main__":
    react_agent("上海今天需要带伞吗?")
