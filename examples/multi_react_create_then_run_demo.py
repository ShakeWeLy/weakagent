"""
Demo: parent follows user prompt to create then run.

Shows:
1) Build AgentRuntime with AgentSpec
2) Create/register a multi_react parent
3) Parent receives user request and calls create_sub_agent
4) Parent then runs the same created sub-agent via run_sub_agent(sub_agent_id)
"""

import asyncio

from weakagent.agent import (
    AgentFactory,
    AgentRuntime,
    AgentSpec,
    BaseAgent,
    BriefReActMultiAgent,
)
from weakagent.schemas.agent import AgentState
from weakagent.tools import ToolCollection, CreateSubAgentTool, RunSubAgentTool, Terminate


class EchoAgent(BaseAgent):
    """Local demo agent to avoid external LLM dependency."""

    async def step(self) -> str:
        latest_user_message = ""
        for msg in reversed(self.messages):
            if msg.role == "user" and msg.content:
                latest_user_message = msg.content
                break
        self.state = AgentState.FINISHED
        return f"{self.name} => {latest_user_message or 'no request'}"


async def main() -> None:
    factory = AgentFactory()
    factory.register_spec(
        "echo",
        AgentSpec(
            agent_cls=EchoAgent,
            default_kwargs={"name": "echo_agent", "max_steps": 1},
            description="Echo demo sub-agent",
        ),
    )

    runtime = await AgentRuntime.instance(factory=factory)
    print(f"Supported runtime agent types: {runtime.factory.supported_types}")

    parent = BriefReActMultiAgent(
        name="parent_multi_react",
        max_steps=3,
        available_tools=ToolCollection(
            CreateSubAgentTool(),
            RunSubAgentTool(),
            Terminate(),
        ),
    )
    parent_id = runtime.register(parent, agent_type="multi_react")
    print(f"Parent created: {parent_id}")

    request = (
        "Please do the following in order:\n"
        "1) Use create_sub_agent to create an 'echo' child with name 'echo_child', "
        "system_prompt 'You are a concise echo child', max_steps 1, and tools ['terminate'].\n"
        "2) Then use run_sub_agent with the returned sub_agent_id (do not create a new one) "
        "to run request 'hello from parent'.\n"
        "3) Return the run result."
    )
    print("\n--- Parent user request ---")
    print(request)

    parent_result = await runtime.run(parent_id, request=request)
    print("\n--- Parent run result ---")
    print(parent_result)
    print(f"\nParent current_sub_agent_id: {parent.current_sub_agent_id}")

    await runtime.cleanup_all()
    print("\nCleanup complete.")


if __name__ == "__main__":
    asyncio.run(main())
