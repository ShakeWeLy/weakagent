from typing import Any, Optional

import json
from pydantic import BaseModel



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