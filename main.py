import asyncio
from typing import List

from pydantic import Field

from weakagent.agent import AgentFactory, AgentRuntime, AgentSpec
from weakagent.agent import ToolCallAgent, BriefReActAgent
from weakagent.tools import ToolCollection, Terminate, AskHumanTool
from weakagent.tools.search import AgentResearchTool
from weakagent.tools import CreateSubAgentTool, RunSubAgentTool
from weakagent.tools.files import GrepTool, ListFilesTool, PatchFileTool, WriteFileTool
from weakagent.schemas.tool import TOOL_CHOICE_TYPE, ToolChoice


# class ChatToolcallAgent(ToolCallAgent):
class ChatToolcallAgent(BriefReActAgent):
    name: str = "chat_toolcall"
    description: str = "A chat agent that can chat with the user and use tools."
    verbose: bool = False
    only_last_result: bool = True


    available_tools: ToolCollection = ToolCollection(
        Terminate(),
        AskHumanTool(),
        AgentResearchTool(),
        CreateSubAgentTool(),
        RunSubAgentTool(),
        GrepTool(),
        ListFilesTool(),
        PatchFileTool(),
        WriteFileTool(),
    )
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO
    special_tool_names: List[str] = Field(
        default_factory=lambda: [Terminate().name, AskHumanTool().name]
    )

    max_steps: int = 100

    async def act(self) -> str:
        return await super().act()

async def main() -> None:
    factory = AgentFactory()
    factory.register_spec(
        "chat_toolcall",
        AgentSpec(
            agent_cls=ChatToolcallAgent,
            default_kwargs={"name": "chat_toolcall_agent",},
            description="Chat agent for runtime demo",
        ),
    )
    runtime = await AgentRuntime.instance(factory=factory)
    chat_id = runtime.create_agent(
        "chat_toolcall",
        name="chat_toolcall_agent",
    )
    await runtime.start_queue_loop(chat_id)

if __name__ == "__main__":
    asyncio.run(main())