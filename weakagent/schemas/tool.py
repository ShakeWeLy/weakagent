'''
你的 ToolCall / Function 本质是：

LLM protocol（协议层）

它不仅 tool 用：

还会被：

memory
logger
agent state
streaming parser
OpenAI adapter

使用, 所以写在schemas里面，而不是tools里面
'''
from typing import Optional

import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel

class ToolChoice(str, Enum):
    """Tool choice options"""

    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


TOOL_CHOICE_VALUES = tuple(choice.value for choice in ToolChoice)
TOOL_CHOICE_TYPE = Literal[TOOL_CHOICE_VALUES]  # type: ignore

class Function(BaseModel):
    name: str
    description: Optional[str] = None
    arguments: str

    def parsed_arguments(self):
        return json.loads(self.arguments)


class ToolCall(BaseModel):
    """Represents a tool/function call in a message"""

    id: str
    type: str = "function"
    function: Function
