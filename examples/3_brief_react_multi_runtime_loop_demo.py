"""
Interactive demo: AgentRuntime + BriefReActMultiAgent multi-turn loop.

Goal:
- Each `agent.run()` uses a fresh `short_memory` (cleared before/after run)
- `runtime_memory` accumulates ONLY request + last_result per run (not cleared)
"""

from __future__ import annotations

import asyncio

from weakagent.agent import AgentRuntime


async def _ainput(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


def _print_runtime_memory(agent) -> None:
    msgs = getattr(agent, "runtime_memory", None)
    if msgs is None:
        print("runtime_memory: <missing>")
        return
    print("\n--- runtime_memory (request + last_result history) ---")
    for i, m in enumerate(msgs.messages, start=1):
        content = (m.content or "").strip().replace("\n", "\\n")
        if len(content) > 160:
            content = content[:160] + "...(truncated)"
        print(f"{i:02d}. role={m.role} content={content}")


async def main() -> None:
    runtime = await AgentRuntime.instance()

    # Create one multi_react agent instance and reuse it across turns.
    agent_id = runtime.create_agent(
        "multi_react",
        name="interactive_multi_react",
        max_steps=6,
    )
    agent = runtime.get(agent_id)

    print("Interactive BriefReActMultiAgent runtime loop.")
    print("Type `exit` to quit.\n")

    try:
        while True:
            user_input = (await _ainput("You> ")).strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q"}:
                break

            result = await runtime.run(agent_id, request=user_input)
            print("\nAgent result:\n", result)
            _print_runtime_memory(agent)
            print()
    finally:
        await runtime.cleanup_all()
        print("Cleanup complete.")


if __name__ == "__main__":
    asyncio.run(main())

