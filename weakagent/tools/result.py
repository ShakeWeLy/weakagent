from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel

from weakagent.schemas.message import Message

class ToolExecutionResult(BaseModel):
    """Result of running a tool (execution layer)."""

    success: bool = True
    output: Optional[str] = None
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

    @classmethod
    def ok(
        cls,
        output: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> ToolExecutionResult:
        return cls(success=True, output=output, data=data)

    @classmethod
    def fail(cls, error: str) -> ToolExecutionResult:
        return cls(success=False, error=error)

    # 写在toolresult 而不是message里面，因为toolresult是工具执行的结果，而message是消息，工具执行的结果需要转换为消息才能发送给用户
    # 后期写成adapter，把toolresult转换为message, 实现双向解耦
    def to_message(
        self,
        tool_call_id: str,
        name: str,
    ) -> Message:
        content = self.output if self.success else self.error
        base64_image = self.data.get("base64_image", None)

        return Message.tool_message(
            content=content or "",
            name=name,
            tool_call_id=tool_call_id,
            base64_image=base64_image
        )

class CLIResult(ToolExecutionResult):
    """Tool result for CLI-like tools."""

    exit_code: int = 0


class ToolFailure(ToolExecutionResult):
    """Explicit failure result."""

    success: bool = False