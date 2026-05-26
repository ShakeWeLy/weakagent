from __future__ import annotations

from typing import TYPE_CHECKING

from weakagent.tools.base import BaseTool, ToolExecutionResult

if TYPE_CHECKING:
    from weakagent.agent.base import BaseAgent

_LIST_TOOLS_DESCRIPTION = (
    "List tools available to this agent: mounted tools and built-in tools "
    "that can still be added via add_tool."
)


class ListToolsTool(BaseTool):
    name: str = "list_tools"
    description: str = _LIST_TOOLS_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {},
    }

    async def execute(self, **kwargs) -> ToolExecutionResult:
        return self.fail_response(
            "list_tools must run inside an agent tool loop (use execute_for_agent)."
        )

    async def execute_for_agent(
        self,
        agent: BaseAgent,
        **kwargs,
    ) -> ToolExecutionResult:
        collection = getattr(agent, "available_tools", None)
        if collection is None:
            return self.fail_response("Agent has no available_tools collection.")

        catalog = collection.list_tool_catalog()
        lines = ["## Mounted tools"]
        mounted = catalog.get("mounted") or []
        if mounted:
            for entry in mounted:
                lines.append(f"- {entry['name']}: {entry['description']}")
        else:
            lines.append("- (none)")

        lines.append("\n## Available to add (built-in)")
        available = catalog.get("available_to_add") or []
        if available:
            for entry in available:
                lines.append(f"- {entry['name']}: {entry['description']}")
        else:
            lines.append("- (none — all discovered built-ins are already mounted)")

        return self.success_response("\n".join(lines))
