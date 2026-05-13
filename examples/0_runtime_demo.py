"""
Demo for BriefReActMultiAgent with run_sub_agent tool.

Shows:
1. Initialize AgentRuntime singleton
2. Create a multi_react parent agent
3. The parent uses run_sub_agent tool to dynamically create and run child agents
4. Cleanup all agents
"""

import asyncio

from weakagent.agent import AgentRuntime, BriefReActMultiAgent
from weakagent.tools import ToolCollection, Terminate
from weakagent.tools.sub_agent import RunSubAgentTool


async def main() -> None:
    # 1. Initialize AgentRuntime singleton
    runtime = await AgentRuntime.instance()
    print(
        f"AgentRuntime initialized. Supported agent types: "
        f"{runtime.factory.supported_types}"
    )

    # 2. Create a multi_react parent agent with run_sub_agent tool
    parent_agent = BriefReActMultiAgent(
        name="parent_multi_agent",
        max_steps=5,
        available_tools=ToolCollection(
            RunSubAgentTool(),
            Terminate(),
        ),
    )

    # Register parent to runtime
    parent_id = runtime.register(parent_agent, agent_type="multi_react")
    print(f"\nParent agent created and registered: {parent_id}")
    print(f"Parent agent has runtime: {parent_agent.agent_runtime is not None}")

    # 3. Run parent agent with a request that should trigger sub-agent creation
    # The LLM should decide to call run_sub_agent tool
    request = (
        "I need to delegate a simple chat task to a sub-agent. "
        "Please use the run_sub_agent tool with agent name 'chat' "
        "and ask it to 'say hello and introduce yourself'."
    )

    print(f"\n--- Running parent agent with request ---")
    print(f"Request: {request}\n")

    try:
        result = await runtime.run(parent_id, request=request)
        print(f"\n--- Parent agent result ---")
        print(result)
    except Exception as e:
        print(f"Error running parent: {e}")

    # Show registered agents after execution
    # print(f"\n--- Registered agents ---")
    # summaries = runtime.get_agent_summaries()
    # for agent_id, info in summaries.items():
    #     print(f"  {agent_id}: {info['name']} - {info['description'][:50] if info['description'] else '(no description)'}")

    # Detailed info
    print(f"\n--- Detailed agent info ---")
    details = runtime.get_registered_agents()
    for info in details:
        print(f"  {info['agent_id']}: type={info['agent_type']}, state={info['state']}, parent={info['parent_id']}, children={info['children']}")

    # 4. Cleanup all agents
    print(f"\n--- Cleanup ---")
    await runtime.cleanup_all()
    print("All agents cleaned up.")

    # Verify cleanup
    remaining = runtime.get_registered_agents()
    print(f"Remaining agents: {len(remaining)}")


if __name__ == "__main__":
    asyncio.run(main())
