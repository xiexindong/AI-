"""
LangGraph 示例拆文件 · 01 · MockLLM
和「简历/Workflow执行引擎示例.py」中的 MockLLM 是同一份逻辑
"""
import json


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

if __name__ == "__main__":
    # 快速自测
    print(llm.chat("任务:判断告警类型"))
    print(llm.chat("任务:自主查证。已有 0 条证据:[] 可选工具:['query_tx', 'query_device']"))
    print(llm.chat("任务:生成研判结论。证据:['流水异常']"))
