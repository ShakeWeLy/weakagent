"""
Demo for AgentRuntime + AgentSpec.

Shows:
1) Register a custom agent type with AgentSpec defaults
2) Create parent/child agents via runtime
3) Run both agents concurrently
4) Cleanup runtime state
"""

import asyncio

from weakagent.agent import AgentFactory, AgentRuntime, AgentSpec, BaseAgent
from weakagent.schemas.agent import AgentState


class EchoAgent(BaseAgent):
    """Minimal agent for runtime orchestration demo."""

    async def step(self) -> str:
        latest_user_message = ""
        for msg in reversed(self.messages):
            if msg.role == "user" and msg.content:
                latest_user_message = msg.content
                break

        self.state = AgentState.FINISHED
        return f"{self.name} handled: {latest_user_message or 'no request'}"


async def main() -> None:
    factory = AgentFactory()
    factory.register_spec(
        "echo",
        AgentSpec(
            agent_cls=EchoAgent,
            default_kwargs={"name": "echo_agent", "max_steps": 1},
            description="Minimal echo agent for runtime demo",
        ),
    )

    runtime = await AgentRuntime.instance(factory=factory)

    parent_id = runtime.create_agent(
        "echo",
        name="parent_echo",
        system_prompt="Parent runtime demo agent",
    )
    child_id = runtime.spawn_sub_agent(
        parent_id=parent_id,
        agent_type="echo",
        name="child_echo",
    )

    print(f"Parent agent id: {parent_id}")
    print(f"Child agent id: {child_id}")
    print(f"Supported runtime agent types: {runtime.factory.supported_types}")
    print(f"Children of parent: {runtime.list_agents(parent_id=parent_id)}")

    parent_task = runtime.run_in_background(parent_id, request="summarize runtime role")
    child_task = runtime.run_in_background(child_id, request="say hello from child")

    results = await asyncio.gather(parent_task, child_task, return_exceptions=True)
    for idx, result in enumerate(results, start=1):
        print(f"\n--- Result {idx} ---")
        print(result)

    await runtime.cleanup_all()
    print("\nCleanup complete.")


if __name__ == "__main__":
    asyncio.run(main())
