import json
from typing import Any, Optional

from weakagent.agent.brief_react import BriefReActAgent
from weakagent.prompt.brief_react import ACT_NEXT_STEP_PROMPT
from weakagent.utils.logger import get_logger
from weakagent.schemas.tool import ToolCall
from weakagent.tools import ToolCollection, Terminate, AskHumanTool
from weakagent.tools.sub_agent.create_sub_agent import CreateSubAgentTool
from weakagent.tools.sub_agent.run_sub_agent import RunSubAgentTool

logger = get_logger(__name__)

ACT_NEXT_STEP_PROMPT = '''
You are a multi-agent reactor that can run sub-agents via tool call.

You will be given a task and a list of sub-agents.
You need to decide which sub-agent to run based on the task.
Sub-agents:
chat: a chat agent that can chat with the user.
echo: an echo agent that can echo the user's message.
react: an agent that git a brieft think, selcet tools to execute tool calls using react.

You need to return the sub-agent id and the request to run the sub-agent, or the answer to the user directly.
If you need to create a new sub-agent, you need to use the create_sub_agent tool.
If you need to run a sub-agent, you need to use the run_sub_agent tool.
If you need to terminate and give a answer to the user directly, you need to use the terminate tool.
'''


class BriefReActMultiAgent(BriefReActAgent):
    """
    Multi-agent variant of BriefReActAgent.

    Uses `run_sub_agent` tool call to delegate current task to a child agent.
    """

    name: str = "multi_react"
    description: str = (
        "A multi-agent reactor that can run sub-agents via tool call."
    )
    # next_step_prompt: str = ACT_NEXT_STEP_PROMPT
    available_tools: ToolCollection = ToolCollection(
        CreateSubAgentTool(), RunSubAgentTool(), Terminate(), AskHumanTool()
    )

    # Using Any to avoid circular import with AgentRuntime
    agent_runtime: Optional[Any] = None
    managed_agent_id: Optional[str] = None
    current_sub_agent_id: Optional[str] = None

    async def execute_tool(self, command: ToolCall) -> str:
        """Intercept sub-agent tools for delegation with parent context."""
        name = command.function.name
        if name not in {RunSubAgentTool.name, CreateSubAgentTool.name}:
            return await super().execute_tool(command)

        try:
            args = json.loads(command.function.arguments or "{}")
        except json.JSONDecodeError:
            return f"Error: Invalid JSON format for {name} arguments"

        # Inject parent context automatically when omitted.
        if "caller_agent_id" not in args or not args.get("caller_agent_id"):
            args["caller_agent_id"] = self.managed_agent_id

        if name == RunSubAgentTool.name:
            if not args.get("sub_agent_id") and not args.get("sub_agent_name"):
                return "Error: run_sub_agent requires `sub_agent_id` or `sub_agent_name`"
            args["request"] = args.get("request") or self._latest_user_request()
        else:
            if not args.get("sub_agent_name"):
                return "Error: create_sub_agent requires `sub_agent_name`"

        switch_result = await self.available_tools.execute(name=name, tool_input=args)
        if not switch_result.success:
            return f"Error: failed to execute {name}: {switch_result.error}"
        return switch_result.output or "Sub-agent executed"

    def _latest_user_request(self) -> str:
        """Get latest user message as fallback delegated request."""
        for msg in reversed(self.messages):
            if msg.role == "user" and msg.content:
                return msg.content
        return ""