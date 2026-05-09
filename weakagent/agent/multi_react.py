import json
from typing import Any, Optional

from weakagent.agent.brief_react import BriefReActAgent
from weakagent.utils.logger import get_logger
from weakagent.schemas.tool import ToolCall
from weakagent.tools import ToolCollection, CreateChatCompletion, Terminate
from weakagent.tools.sub_agent.run_sub_agent import RunSubAgentTool

logger = get_logger(__name__)

class BriefReActMultiAgent(BriefReActAgent):
    """
    Multi-agent variant of BriefReActAgent.

    Uses `run_sub_agent` tool call to delegate current task to a child agent.
    """

    name: str = "multi_react"
    description: str = (
        "A multi-agent reactor that can run sub-agents via tool call."
    )
    available_tools: ToolCollection = ToolCollection(
        CreateChatCompletion(), RunSubAgentTool(), Terminate()
    )

    # Using Any to avoid circular import with AgentRuntime
    agent_runtime: Optional[Any] = None
    # Backward-compatible alias field; still supported in runtime wiring.
    agent_manager: Optional[Any] = None
    managed_agent_id: Optional[str] = None
    current_sub_agent_id: Optional[str] = None

    async def execute_tool(self, command: ToolCall) -> str:
        """Intercept `run_sub_agent` for delegation."""
        name = command.function.name
        if name != RunSubAgentTool.name:
            return await super().execute_tool(command)

        try:
            args = json.loads(command.function.arguments or "{}")
        except json.JSONDecodeError:
            return "Error: Invalid JSON format for run_sub_agent arguments"

        sub_agent_name = args.get("sub_agent_name")
        request = args.get("request") or self._latest_user_request()
        if not sub_agent_name:
            return "Error: run_sub_agent requires `sub_agent_name`"

        switch_result = await self.available_tools.execute(
            name=RunSubAgentTool.name,
            tool_input={
                "sub_agent_name": sub_agent_name,
                "request": request,
                "caller_agent_id": self.managed_agent_id,
            },
        )
        if not switch_result.success:
            return f"Error: failed to run sub-agent: {switch_result.error}"
        return switch_result.output or "Sub-agent executed"

    def _latest_user_request(self) -> str:
        """Get latest user message as fallback delegated request."""
        for msg in reversed(self.messages):
            if msg.role == "user" and msg.content:
                return msg.content
        return ""


# Backward-compatible alias for existing typo name.
BriefReAcMultiAgent = BriefReActMultiAgent