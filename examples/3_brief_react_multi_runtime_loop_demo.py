"""
Interactive demo: AgentRuntime.run_loop + BriefReActMultiAgent.

Goal:
- Each `agent.run()` uses per-run `short_memory` (working context per turn)
- `session` / conversation persist the full transcript across turns (sqlite)

Prerequisites:
  - config.toml: [llm.fast]

Run from project root:

    python examples/3_brief_react_multi_runtime_loop_demo.py
    python examples/3_brief_react_multi_runtime_loop_demo.py --request "create an echo sub-agent"
    python examples/3_brief_react_multi_runtime_loop_demo.py --load-last-session
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from weakagent.agent import AgentRuntime
from weakagent.llm.llm import LLM
from weakagent.memory.base import MemoryType
from weakagent.memory.conversation import ConversationMemory
from weakagent.memory.session import SessionMemory
from weakagent.utils.logger import logger

STABLE_AGENT_ID = "multi_react_loop_demo"


def _print_session_transcript(agent_id: str, *, db_path: Optional[str] = None) -> None:
    """Print the latest closed session transcript for this demo agent."""
    path = db_path or str(
        SessionMemory._resolve_db_path("weakagent.sqlite3", MemoryType.SESSION)
    )
    session_id = SessionMemory.get_last_session_id_for_agent(agent_id, db_path=path)
    if not session_id:
        print("\n(session transcript: none)")
        return

    messages = ConversationMemory.fetch_messages(
        db_path=path,
        session_id=session_id,
        agent_id=agent_id,
    )
    print(f"\n--- session transcript (session_id={session_id}) ---")
    if not messages:
        print("  (empty)")
        return
    for i, m in enumerate(messages, start=1):
        content = (m.content or "").strip().replace("\n", "\\n")
        if len(content) > 160:
            content = content[:160] + "...(truncated)"
        print(f"{i:02d}. role={m.role} content={content}")


async def run_interactive_loop(
    *,
    first_request: Optional[str] = None,
    load_last_session: bool = False,
    last_session_messages: int = 10,
    use_long_memory: bool = False,
    user_id: Optional[str] = "demo_user",
    show_session: bool = True,
) -> None:
    runtime = await AgentRuntime.instance()
    agent_id = runtime.create_agent(
        "multi_react",
        agent_id=STABLE_AGENT_ID,
        name="interactive_multi_react",
        llm=LLM(config_name="fast"),
        max_steps=6,
        summarize_short_memory=True,
        user_id=user_id,
        use_long_memory=use_long_memory,
    )

    print("Interactive BriefReActMultiAgent (AgentRuntime.run_loop).")
    print("Type exit / quit / q to end.\n")

    try:
        await runtime.run_loop(
            agent_id,
            request=first_request,
            load_last_session=load_last_session,
            last_session_messages=last_session_messages,
        )
    except Exception:
        logger.exception("Failed to run interactive loop")
    finally:
        if show_session:
            _print_session_transcript(STABLE_AGENT_ID)
        AgentRuntime._instance = None


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="AgentRuntime.run_loop + BriefReActMultiAgent",
    )
    parser.add_argument(
        "--request",
        default=None,
        help="Optional first user message before interactive prompts",
    )
    parser.add_argument(
        "--load-last-session",
        action="store_true",
        help="Load raw messages from the previous runtime session (default: last 10)",
    )
    parser.add_argument(
        "--last-session-messages",
        type=int,
        default=10,
        help="How many raw messages to load (-1 = all). Used with --load-last-session",
    )
    parser.add_argument(
        "--use-long-memory",
        action="store_true",
        help="Inject long memory each run; extract from session on loop exit",
    )
    parser.add_argument(
        "--user-id",
        default="demo_user",
        help="agent.user_id for session / conversation / long memory",
    )
    parser.add_argument(
        "--no-show-session",
        action="store_true",
        help="Skip printing session transcript after the loop ends",
    )
    args = parser.parse_args()

    try:
        await run_interactive_loop(
            first_request=args.request,
            load_last_session=args.load_last_session,
            last_session_messages=args.last_session_messages,
            use_long_memory=args.use_long_memory,
            user_id=args.user_id or None,
            show_session=not args.no_show_session,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
