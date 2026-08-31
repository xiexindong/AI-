"""
LangGraph 示例拆文件 · 02 · 风控工具函数
对应「简历/Workflow执行引擎示例.py」里的 ToolNode 取数逻辑
"""


def query_tx(user_id: str) -> str:
    """查询交易流水"""
    return "12 笔共 63 万,其中 3 笔深夜大额"


def query_device(user_id: str) -> str:
    """查询设备指纹"""
    return "24h 内登录设备 2 台 → 5 台,新增 3 台 rooted"


def query_profile(user_id: str) -> str:
    """查询用户画像"""
    return "历史月均交易 8 万"


if __name__ == "__main__":
    print(query_tx("U1001"))
    print(query_device("U1001"))
    print(query_profile("U1001"))
