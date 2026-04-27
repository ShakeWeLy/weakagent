from typing import Any, Optional

import json
from pydantic import BaseModel
from typing import Literal
from enum import Enum

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


class ToolResult(BaseModel):
    tool_call_id: str
    name: str
    result: Any
    success: bool = True
    error: Optional[str] = None