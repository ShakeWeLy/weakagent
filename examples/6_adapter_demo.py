"""
Demo for input/output adapters with AgentRuntime.

Shows:
1) runtime.run() with APIOutput (programmatic request, no CLI)
2) runtime.run_loop() with APIInput / APIOutput (multi-turn, then exit)
3) runtime.run_in_background() with emit_output to APIOutput

Run from project root:

    python examples/6_adapter_demo.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from weakagent.adapters.input import APIInput
from weakagent.adapters.output import APIOutput
from weakagent.agent import AgentFactory, AgentRuntime, AgentSpec, BaseAgent
from weakagent.schemas.agent import AgentState
from weakagent.utils.logger import logger

LOOP_AGENT_ID = "adapter_echo_loop_demo"


class EchoAgent(BaseAgent):
    """Minimal agent; no LLM required."""

    def model_post_init(self, __context: object) -> None:
        # BaseAgent auto-creates session/conversation; disable for offline demo.
        self.session = None
        self.conversation = None

    async def step(self) -> str:
        latest_user_message = ""
        for msg in reversed(self.messages):
            if msg.role == "user" and msg.content:
                latest_user_message = msg.content
                break
        self.state = AgentState.FINISHED
        return f"{self.name} <= {latest_user_message or '(empty)'}"


async def demo_run_with_api_output(runtime: AgentRuntime) -> None:
    """Part 1: single run — request in code, result via APIOutput queue."""
    print("\n=== Part 1: run() + APIOutput ===")
    agent_id = runtime.create_agent("echo", name="adapter_echo")
    result_queue: asyncio.Queue[str] = asyncio.Queue()
    try:
        result = await runtime.run(
            agent_id,
            request="ping from adapter demo",
            output_source=APIOutput(result_queue),
            emit_output=True,
            load_last_session=False,
        )
        queued = await result_queue.get()
        print(f"return value : {result[:120]}...")
        print(f"queue output : {queued[:120]}...")
    finally:
        if agent_id in runtime._agents:
            await runtime.cleanup(agent_id)


async def demo_run_loop_api(runtime: AgentRuntime) -> None:
    """Part 2: run_loop — feed requests through APIInput, collect APIOutput."""
    print("\n=== Part 2: run_loop() + APIInput / APIOutput ===")
    agent_id = runtime.create_agent(
        "echo",
        agent_id=LOOP_AGENT_ID,
        name="adapter_echo_loop",
    )
    request_queue: asyncio.Queue[str] = asyncio.Queue()
    result_queue: asyncio.Queue[str] = asyncio.Queue()

    async def feed_requests() -> None:
        await request_queue.put("first api turn")
        await request_queue.put("second api turn")
        await request_queue.put("exit")

    loop_task = asyncio.create_task(
        runtime.run_loop(
            agent_id,
            input_source=APIInput(request_queue),
            output_source=APIOutput(result_queue),
            load_last_session=False,
        )
    )
    feeder = asyncio.create_task(feed_requests())

    results: list[str] = []
    while len(results) < 2:
        results.append(await result_queue.get())

    await feeder
    await loop_task

    for idx, text in enumerate(results, start=1):
        print(f"turn {idx}: {text[:120]}...")


async def demo_run_in_background_api(runtime: AgentRuntime) -> None:
    """Part 3: background task writes to APIOutput; await Task for return value."""
    print("\n=== Part 3: run_in_background() + APIOutput ===")
    agent_id = runtime.create_agent("echo", name="adapter_echo_bg")
    result_queue: asyncio.Queue[str] = asyncio.Queue()

    try:
        task = runtime.run_in_background(
            agent_id,
            request="background adapter message",
            output_source=APIOutput(result_queue),
            emit_output=True,
            load_last_session=False,
        )
        result = await task
        queued = await result_queue.get()
        print(f"task result  : {result[:120]}...")
        print(f"queue output : {queued[:120]}...")
    finally:
        if agent_id in runtime._agents:
            await runtime.cleanup(agent_id)


async def main() -> None:
    factory = AgentFactory()
    factory.register_spec(
        "echo",
        AgentSpec(
            agent_cls=EchoAgent,
            default_kwargs={"name": "echo_agent", "max_steps": 1},
            description="Echo agent for adapter demo",
        ),
    )

    try:
        runtime = await AgentRuntime.instance(factory=factory)
        await demo_run_with_api_output(runtime)
        await demo_run_loop_api(runtime)
        await demo_run_in_background_api(runtime)
        print("\nAdapter demo complete.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        logger.exception("Adapter demo failed")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        AgentRuntime._instance = None


if __name__ == "__main__":
    asyncio.run(main())
