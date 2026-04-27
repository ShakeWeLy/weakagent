from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, Field

from .result import ToolExecutionResult

# =========================================================
# Base Tool
# =========================================================
class BaseTool(ABC):
    """
    Base class for all tools
    """

    name: str = ""
    description: str = ""
    args_model: Type[BaseModel] = BaseModel

    # optional metadata
    timeout: int = 30
    retry: int = 0
    cost: str = "low"

    async def __call__(self, **kwargs) -> ToolExecutionResult:
        """
        Validate args and execute tool
        """
        try:
            args = self.args_model(**kwargs)
            return await self.execute(args)
        except Exception as e:
            return self.fail_response(str(e))

    @abstractmethod
    async def execute(self, args: BaseModel) -> ToolExecutionResult:
        """
        Subclass must implement
        """
        raise NotImplementedError

    def to_tool_schema(self) -> dict:
        """
        OpenAI tool calling schema
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
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
# Example Tool
# =========================================================
class SearchArgs(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, ge=1, le=10)


class SearchTool(BaseTool):
    name = "search"
    description = "Search information from internet"
    args_model = SearchArgs

    async def execute(self, args: SearchArgs) -> ToolExecutionResult:
        try:
            result = {
                "query": args.query,
                "top_k": args.top_k,
                "items": [
                    f"Result 1 for {args.query}",
                    f"Result 2 for {args.query}",
                ],
            }
            return self.success_response(result)

        except Exception as e:
            return self.fail_response(str(e))


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