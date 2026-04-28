"""项目内业务异常。"""


class TokenLimitExceeded(Exception):
    """在累计或单次请求会超过已配置的 input token 上限时抛出。"""


class ModelCapabilityError(Exception):
    """当前 LLM 配置不具备所需能力时抛出（例如未开启多模态却调用带图 API）。"""

class ToolError(Exception):
    """在工具执行过程中抛出。"""
