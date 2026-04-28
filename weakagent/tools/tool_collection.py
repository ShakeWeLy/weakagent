"""Collection classes for managing multiple tools."""

from typing import Any, Dict, List

from weakagent.utils.exceptions import ToolError
from weakagent.utils.logger import get_logger
from weakagent.tools.base import BaseTool, ToolExecutionResult

logger = get_logger(__name__)

class ToolCollection:
    """A collection of defined tools."""

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, *tools: BaseTool):
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}

    def __iter__(self):
        return iter(self.tools)

    def to_params(self) -> List[Dict[str, Any]]:
        return [tool.to_params() for tool in self.tools]

    async def execute(
        self, *, name: str, tool_input: Dict[str, Any] = None
    ) -> ToolExecutionResult:
        tool = self.tool_map.get(name)
        if not tool:
            return ToolExecutionResult.fail(error=f"Tool {name} is invalid")
        try:
            result = await tool(**tool_input)
            return result
        except ToolError as e:
            return ToolExecutionResult.fail(error=str(e))

    async def execute_all(self) -> List[ToolExecutionResult]:
        """Execute all tools in the collection sequentially."""
        results = []
        for tool in self.tools:
            try:
                result = await tool()
                results.append(result)
            except ToolError as e:
                results.append(ToolExecutionResult.fail(error=str(e)))
        return results

    def get_tool(self, name: str) -> BaseTool:
        return self.tool_map.get(name)

    def add_tool(self, tool: BaseTool):
        """Add a single tool to the collection.

        If a tool with the same name already exists, it will be skipped and a warning will be logged.
        """
        if tool.name in self.tool_map:
            logger.warning(f"Tool {tool.name} already exists in collection, skipping")
            return self

        self.tools += (tool,)
        self.tool_map[tool.name] = tool
        return self

    def add_tools(self, *tools: BaseTool):
        """Add multiple tools to the collection.

        If any tool has a name conflict with an existing tool, it will be skipped and a warning will be logged.
        """
        for tool in tools:
            self.add_tool(tool)
        return self
