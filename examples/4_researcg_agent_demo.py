"""
Demo for AgentRuntime + AgentSpec.

Shows:
1) Register a custom agent type with AgentSpec defaults
2) Create a research agent and run it
"""

import asyncio

from weakagent.agent import AgentFactory, AgentRuntime, AgentSpec, ResearchAgent



async def main() -> None:
    factory = AgentFactory()
    factory.register_spec(
        "research",
        AgentSpec(
            agent_cls=ResearchAgent,
            default_kwargs={"name": "research_agent", "max_steps":10},
            description="Research agent for runtime demo",
        ),
    )

    runtime = await AgentRuntime.instance(factory=factory)

    research_id = runtime.create_agent(
        "research",
        name="research_agent",
        system_prompt="Research agent for runtime demo",
    )
    request = input("Enter a request to research: ")
    # request = "What is the latest news on AI?"
    results = await runtime.run(research_id, request=request)
    print(results)

    await runtime.cleanup_all()
    print("Cleanup complete.")


if __name__ == "__main__":
    asyncio.run(main())
