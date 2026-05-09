"""
Demo for AgentFactory + AgentRuntime.

Shows:
1) Create a parent agent
2) Spawn a sub agent from the parent
3) Run both agents concurrently
4) Cleanup managed agents/resources
"""

import asyncio

from weakagent.agent import AgentRuntime


async def main() -> None:
    runtime = await AgentRuntime.instance()

    # 1) Create parent agent
    parent_id = runtime.create_agent(
        "chat",
        name="parent_chat_agent",
        config_name="default",
    )

    # 2) Spawn sub agent
    child_id = runtime.spawn_sub_agent(
        parent_id=parent_id,
        agent_type="chat",
        name="child_chat_agent",
        config_name="default",
    )

    print(f"Parent agent id: {parent_id}")
    print(f"Child  agent id: {child_id}")
    print(f"Children of parent: {runtime.list_agents(parent_id=parent_id)}")

    # 3) Run parent + child concurrently
    parent_task = runtime.run_in_background(
        parent_id, request="Please summarize what an agent runtime does."
    )
    child_task = runtime.run_in_background(
        child_id, request="Please give 3 short tips for sub-agent orchestration."
    )

    results = await asyncio.gather(parent_task, child_task, return_exceptions=True)
    for idx, result in enumerate(results, start=1):
        print(f"\n--- Result {idx} ---")
        if isinstance(result, Exception):
            print(f"Error: {result}")
        else:
            print(result)

    # 4) Cleanup all agents/resources/tasks
    await runtime.cleanup_all()
    print("\nCleanup complete.")


if __name__ == "__main__":
    asyncio.run(main())

