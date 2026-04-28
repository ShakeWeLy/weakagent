from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, Field

from weakagent.schemas.message import Message

# =========================================================
# Tool Execution Result
# =========================================================
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
    # def to_message(
    #     self,
    #     tool_call_id: str,
    #     name: str,
    # ) -> Message:
    #     content = self.output if self.success else self.error
    #     base64_image = self.data.get("base64_image", None)

    #     return Message.tool_message(
    #         content=content or "",
    #         name=name,
    #         tool_call_id=tool_call_id,
    #         base64_image=base64_image
    #     )

class CLIExecutionResult(ToolExecutionResult):
    """Result of running a CLI tool (execution layer)."""


# =========================================================
# Base Tool
# =========================================================
class BaseTool(ABC):
    """
    Base class for all tools
    """

    name: str = ""
    description: str = ""
    args_model: Optional[Type[BaseModel]] = None
    parameters: dict = Field(default_factory=dict, description="Parameters for the tool")

    # optional metadata
    timeout: int = 30
    retry: int = 0
    cost: str = "low"

    async def __call__(self, **kwargs) -> ToolExecutionResult:
        """
        Validate args and execute tool
        """
        try:
            if self.args_model is not None:
                validated_args = self.args_model(**kwargs)
                return await self.execute(validated_args)
            return await self.execute(**kwargs)
        except Exception as e:
            return self.fail_response(str(e))

    @abstractmethod
    async def execute(self, *args, **kwargs) -> ToolExecutionResult:
        """
        Subclass must implement
        """
        raise NotImplementedError

    def to_params(self) -> dict:
        """
        OpenAI tool calling parameters
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def success_response(
        self,
        data: str | Dict[str, Any],
    ) -> ToolExecutionResult:
        """
        Build success result
        """
        if isinstance(data, str):
            text = data
            raw = None
        else:
            text = json.dumps(data, indent=2, ensure_ascii=False)
            raw = data

        return ToolExecutionResult.ok(output=text, data=raw)

    def fail_response(self, msg: str) -> ToolExecutionResult:
        """
        Build failed result
        """
        return ToolExecutionResult.fail(error=msg)


# =========================================================
# Tool Registry
# =========================================================
class ToolRegistry:
    """
    Register and dispatch tools
    """

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        self.tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found")
        return self.tools[name]

    def schemas(self) -> list[dict]:
        return [tool.to_tool_schema() for tool in self.tools.values()]

    async def execute(self, name: str, arguments: dict) -> ToolExecutionResult:
        tool = self.get(name)
        return await tool(**arguments)


# =========================================================
# Example Usage
# =========================================================
"""
import asyncio

async def main():
    registry = ToolRegistry()

    registry.register(SearchTool())

    print(registry.schemas())

    result = await registry.execute(
        "search",
        {
            "query": "Python Agent Framework",
            "top_k": 3
        }
    )

    print(result.model_dump())

asyncio.run(main())
"""