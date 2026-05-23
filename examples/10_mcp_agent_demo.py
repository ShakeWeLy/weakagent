"""
Runtime loop + Skills + MCP combined demo.

Flow:
  skills/ + workspace/skills/  -> <available_skills> in act system prompt
  config.toml [mcp]            -> MCP tools merged into agent
  AgentRuntime.run_loop()      -> interactive You> loop, runtime_memory persists

Prerequisites:
  - config.toml: [llm.fast], [skills], [mcp] (Record App on :3001 if using MCP)
  - Optional: skills/demo-skill/SKILL.md

Run from project root:

    python examples/10_mcp_agent_demo.py
    python examples/10_mcp_agent_demo.py --request "列出待办"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import Field

from weakagent.agent import AgentFactory, AgentRuntime, AgentSpec, BriefReActAgent, ToolCallAgent
from weakagent.llm.llm import LLM
from weakagent.mcp import attach_mcp_to_tools, load_mcp_settings
from weakagent.prompt.brief_react import ACT_SYSTEM_PROMPT, THINK_SYSTEM_PROMPT
from weakagent.schemas.tool import TOOL_CHOICE_TYPE, ToolChoice
from weakagent.skills.manager import SkillManager
from weakagent.tools import ToolCollection, Terminate
from weakagent.tools.files import ReadFileTool

SYSTEM_PROMPT = f"""## Skills
When the user task matches an entry in <available_skills>, use the `read` tool to load
the skill file at <location> (usually SKILL.md), then follow its instructions.

## MCP (Record App)
Remote tools are prefixed with `mcp_record-app_` (or similar). Use them for todos/checkins:
- list_todos, add_todo, complete_todo, list_checkins, record_checkin
Read current state with list_* before mutating. Do not invent data.
"""

COMBINED_ACT_PROMPT = f"""{ACT_SYSTEM_PROMPT}+{SYSTEM_PROMPT}
"""

class RuntimeSkillsMCPAgent(BriefReActAgent):
    """Brief ReAct agent: runtime loop + skills catalog + MCP tools."""

    name: str = "runtime_skills_mcp"
    description: str = "Interactive agent with skills and external MCP"
    think_system_prompt: str = THINK_SYSTEM_PROMPT
    act_system_prompt: str = COMBINED_ACT_PROMPT
    available_tools: ToolCollection = ToolCollection(Terminate(), ReadFileTool())
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])
    skills_enabled: bool = True
    max_steps: int = 12
    verbose: bool = True
    only_last_result: bool = True

# class RuntimeSkillsMCPAgent(ToolCallAgent):
#     name: str = "runtime_skills_mcp"
#     description: str = "Interactive agent with skills and external MCP"
#     system_prompt: str = SYSTEM_PROMPT
#     available_tools: ToolCollection = ToolCollection(Terminate(), ReadFileTool())
#     tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore
#     special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])
#     skills_enabled: bool = True
#     max_steps: int = 12
#     verbose: bool = False
#     only_last_result: bool = True

def _print_skills_catalog() -> None:
    mgr = SkillManager()
    entries = mgr.filter_skills()
    print("=== Skills (enabled) ===")
    if not entries:
        print("  (none — add skills under skills/ or workspace/skills/)")
        return
    for e in entries:
        print(f"  - {e.skill.name}: {e.skill.description[:70]}")
        print(f"    file: {e.skill.file_path}")


def _print_mcp_tools(agent: RuntimeSkillsMCPAgent) -> None:
    print("=== MCP tools ===")
    found = False
    for name, tool in agent.available_tools.tool_map.items():
        if not name.startswith("mcp_"):
            continue
        found = True
        orig = getattr(tool, "original_name", "?")
        sid = getattr(tool, "server_id", "?")
        print(f"  - {name} (server={sid}, mcp={orig})")
    if not found:
        print("  (none — enable [mcp] in config.toml and start the MCP server)")


async def run_interactive_loop(
    *,
    first_request: Optional[str] = None,
    load_last_session: bool = False,
    last_session_messages: int = 10,
    use_long_memory: bool = False,
    long_memory_user_id: Optional[str] = "demo_user",
) -> None:
    mcp_settings = load_mcp_settings()

    factory = AgentFactory()
    factory.register_spec(
        "runtime_skills_mcp",
        AgentSpec(
            agent_cls=RuntimeSkillsMCPAgent,
            default_kwargs={"name": "runtime_skills_mcp_agent"},
            description="Runtime loop with skills + MCP",
        ),
    )

    runtime = await AgentRuntime.instance(factory=factory)
    # Stable agent_id so --load-last-session can find prior loop sessions in sqlite.
    stable_agent_id = "runtime_skills_mcp_demo"
    agent_id = runtime.create_agent(
        "runtime_skills_mcp",
        agent_id=stable_agent_id,
        name="runtime_skills_mcp_agent",
        llm=LLM(config_name="fast"),
    )
    agent = runtime.get(agent_id)
    assert isinstance(agent, RuntimeSkillsMCPAgent)
    if long_memory_user_id:
        agent.long_memory_user_id = long_memory_user_id
        agent.runtime_memory.user_id = long_memory_user_id
        agent.runtime_long_memory.user_id = long_memory_user_id
        agent.runtime_memory.ensure_session()
        agent.refresh_long_memory()

    mcp_clients = None
    if mcp_settings.enabled:
        _, mcp_clients = await attach_mcp_to_tools(agent.available_tools)

    _print_skills_catalog()
    _print_mcp_tools(agent)

    print("\n=== Runtime loop (type exit / quit / q to end) ===")
    print("Layers: runtime_memory (跨轮) | skills (<available_skills>) | MCP tools\n")

    try:
        await runtime.run_loop(
            agent_id,
            request=first_request,
            load_last_session=load_last_session,
            last_session_messages=last_session_messages,
            use_long_memory=use_long_memory,
        )
    finally:
        if mcp_clients:
            await mcp_clients.disconnect()
        # run_loop already cleanup(agent_id); reset singleton for clean re-run in tests
        AgentRuntime._instance = None


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="AgentRuntime.run_loop + skills + MCP",
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
        help="Inject long memory each run; extract from runtime_memory on loop exit",
    )
    parser.add_argument(
        "--long-memory-user-id",
        default="demo_user",
        help="User id for long_term_memory sqlite rows",
    )
    args = parser.parse_args()

    try:
        await run_interactive_loop(
            first_request=args.request,
            load_last_session=args.load_last_session,
            last_session_messages=args.last_session_messages,
            use_long_memory=args.use_long_memory,
            long_memory_user_id=args.long_memory_user_id or None,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
