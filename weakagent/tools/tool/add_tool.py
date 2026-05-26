from __future__ import annotations

from typing import TYPE_CHECKING

from weakagent.tools.base import BaseTool, ToolExecutionResult

if TYPE_CHECKING:
    from weakagent.agent.base import BaseAgent

_ADD_TOOL_DESCRIPTION = (
    "Add a built-in tool to the current agent by name. "
    "Use list_tools first to see tool names that are not yet mounted."
)


class AddToolTool(BaseTool):
    name: str = "add_tool"
    description: str = _ADD_TOOL_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "description": "Built-in tool name to mount (from list_tools).",
            }
        },
        "required": ["tool_name"],
    }

    async def execute(self, tool_name: str, **kwargs) -> ToolExecutionResult:
        return self.fail_response(
            "add_tool must run inside an agent tool loop (use execute_for_agent)."
        )

    async def execute_for_agent(
        self,
        agent: BaseAgent,
        *,
        tool_name: str,
        **kwargs,
    ) -> ToolExecutionResult:
        name = (tool_name or "").strip()
        if not name:
            return self.fail_response("`tool_name` must be a non-empty string.")

        collection = getattr(agent, "available_tools", None)
        if collection is None:
            return self.fail_response("Agent has no available_tools collection.")

        if name in collection.tool_map:
            return self.success_response(f"Tool '{name}' is already mounted.")

        try:
            added = await agent.add_tool_dynamically(name)
        except ValueError as exc:
            return self.fail_response(str(exc))

        return self.success_response(
            f"Tool '{added.name}' added and is now available for tool calls."
        )
