"""
Full demo: Runtime loop + Skills + MCP + Scheduler + Adapters (CLI I/O).

Flow:
  skills/ + workspace/skills/  -> <available_skills> in system prompt
  config.toml [mcp]            -> MCP tools merged into agent
  TaskStore + SchedulerRunner  -> background scan/dispatch for due tasks
  create_task / list_tasks / … -> agent CRUD on the shared TaskStore
  CLIInput / CLIOutput         -> interactive You> loop via adapters
  AgentRuntime.run_loop()      -> session transcript persists

Prerequisites:
  - config.toml: [llm.fast], [skills], [mcp] (Record App on :3001 if using MCP)
  - Optional: [scheduler].db_path in config.toml (defaults to tasks.sqlite3)
  - Optional: skills/demo-skill/SKILL.md

Run from project root:

    python examples/11_full_demo.py
    python examples/11_full_demo.py --request "列出待办"
    python examples/11_full_demo.py --load-last-session
    python examples/11_full_demo.py --no-scheduler
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

from weakagent.adapters.input import CLIInput
from weakagent.adapters.output import CLIOutput
from weakagent.agent import AgentFactory, AgentRuntime, AgentSpec, ToolCallAgent
from weakagent.llm.llm import LLM
from weakagent.mcp import MCPClients, attach_mcp_to_tools, load_mcp_settings
from weakagent.prompt.toolcall import SYSTEM_PROMPT as TOOLCALL_SYSTEM_PROMPT
from weakagent.scheduler import (
    Dispatcher,
    Scheduler,
    SchedulerRunner,
    TaskRegistry,
    TaskStatus,
    TaskStore,
)
from weakagent.scheduler.executors import DailySummaryExecutor, WeeklyReportExecutor
from weakagent.schemas.tool import TOOL_CHOICE_TYPE, ToolChoice
from weakagent.skills.manager import SkillManager
from weakagent.tools import ToolCollection, Terminate
from weakagent.tools.scheduler import (
    CreateTaskTool,
    DeleteTaskTool,
    GetTaskTool,
    ListTasksTool,
    UpdateTaskTool,
)
from weakagent.tools.scheduler._common import set_task_store
from weakagent.tools.files import (
    GrepTool,
    ListFilesTool,
    PatchFileTool,
    ReadFileTool,
    WriteFileTool,
)
from weakagent.tools.search import AgentResearchTool
from weakagent.tools.special_tool.ask_human import AskHumanTool
from weakagent.tools.sub_agent import CreateSubAgentTool, RunSubAgentTool
from weakagent.utils.logger import logger

EXTRA_SYSTEM_PROMPT = """## Skills
When the user task matches an entry in <available_skills>, use the `read` tool to load
the skill file at <location> (usually SKILL.md), then follow its instructions.

## MCP (Record App)
Remote tools are prefixed with `mcp_record-app_` (or similar). Use them for todos/checkins:
- list_todos, add_todo, complete_todo, list_checkins, record_checkin
Read current state with list_* before mutating. Do not invent data.

## Scheduler
Use scheduler tools to manage deferred/periodic work in TaskStore:
- create_task, list_tasks, get_task, update_task, delete_task
Registered task_type values: `daily_summary`, `weekly_report` (payload is JSON object).
A background SchedulerRunner dispatches due pending tasks automatically.
"""

COMBINED_SYSTEM_PROMPT = f"""{TOOLCALL_SYSTEM_PROMPT.rstrip()}

{EXTRA_SYSTEM_PROMPT.strip()}
"""


def build_scheduler_stack(store: TaskStore) -> Scheduler:
    """Wire TaskRegistry + Dispatcher + Scheduler (see examples/5_scheduler_demo.py)."""
    registry = TaskRegistry()
    registry.register("daily_summary", DailySummaryExecutor)
    registry.register("weekly_report", WeeklyReportExecutor)
    dispatcher = Dispatcher(registry=registry, store=store)
    return Scheduler(store=store, dispatcher=dispatcher)


class FullDemoAgent(ToolCallAgent):
    """Tool-calling agent: runtime loop + skills + MCP + scheduler + local tools."""

    name: str = "full_demo"
    description: str = "Full demo agent with skills, MCP, scheduler, and file/sub-agent tools"
    system_prompt: str = COMBINED_SYSTEM_PROMPT
    available_tools: ToolCollection = ToolCollection(
        Terminate(),
        ReadFileTool(),
        AskHumanTool(),
        AgentResearchTool(),
        CreateSubAgentTool(),
        RunSubAgentTool(),
        GrepTool(),
        ListFilesTool(),
        PatchFileTool(),
        WriteFileTool(),
        CreateTaskTool(),
        ListTasksTool(),
        GetTaskTool(),
        UpdateTaskTool(),
        DeleteTaskTool(),
    )
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore
    special_tool_names: List[str] = Field(
        default_factory=lambda: [Terminate().name, AskHumanTool().name]
    )
    skills_enabled: bool = True
    max_steps: int = 100
    verbose: bool = True
    only_last_result: bool = True


def _print_skills_catalog() -> None:
    """List enabled skills from builtin + workspace directories."""
    mgr = SkillManager()
    entries = mgr.filter_skills()
    print("=== Skills (enabled) ===")
    if not entries:
        print("  (none — add skills under skills/ or workspace/skills/)")
        return
    for e in entries:
        desc = (e.skill.description or "")[:70]
        print(f"  - {e.skill.name}: {desc}")
        print(f"    file: {e.skill.file_path}")


def _print_scheduler_info(store: TaskStore, *, runner_active: bool) -> None:
    """Show TaskStore path, runner state, and pending task count."""
    print("=== Scheduler ===")
    print(f"  db_path : {store.db_path}")
    print(f"  runner  : {'on (background scan)' if runner_active else 'off'}")
    pending = [t for t in store.list_tasks() if t.status == TaskStatus.PENDING]
    print(f"  pending : {len(pending)} task(s)")
    for t in pending[:5]:
        print(f"    - id={t.id} type={t.task_type} next_run_at={t.next_run_at}")
    if len(pending) > 5:
        print(f"    ... and {len(pending) - 5} more")


def _print_mcp_tools(agent: FullDemoAgent) -> None:
    """Show MCP tools merged into the agent (names are prefixed with mcp_<server>_)"""
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
    load_last_session: bool = True,
    last_session_messages: int = 10,
    use_long_memory: bool = True,
    user_id: Optional[str] = "demo_user",
    scheduler_enabled: bool = True,
    scheduler_interval: float = 5.0,
) -> None:
    """Create agent, attach MCP + scheduler, then run the CLI adapter loop until exit/quit/q."""
    mcp_settings = load_mcp_settings()
    print(f"MCP enabled: {mcp_settings.enabled}")

    store = TaskStore()
    set_task_store(store)
    scheduler = build_scheduler_stack(store)
    runner: Optional[SchedulerRunner] = None
    if scheduler_enabled:
        runner = SchedulerRunner(scheduler, interval=scheduler_interval)
        runner.start()

    factory = AgentFactory()
    factory.register_spec(
        "full_demo",
        AgentSpec(
            agent_cls=FullDemoAgent,
            default_kwargs={"name": "full_demo_agent"},
            description="Full demo: skills + MCP + adapters + tool calling",
        ),
    )

    runtime = await AgentRuntime.instance(factory=factory)
    stable_agent_id = "full_demo"
    agent_id = runtime.create_agent(
        "full_demo",
        agent_id=stable_agent_id,
        name="full_demo_agent",
        llm=LLM(config_name="fast"),
        user_id=user_id,
        use_long_memory=use_long_memory,
    )
    agent = runtime.get(agent_id)
    assert isinstance(agent, FullDemoAgent)

    mcp_clients: Optional[MCPClients] = None
    if mcp_settings.enabled:
        _, mcp_clients = await attach_mcp_to_tools(agent.available_tools)

    _print_skills_catalog()
    _print_mcp_tools(agent)
    _print_scheduler_info(store, runner_active=runner is not None)

    print("\n=== Runtime loop (type exit / quit / q to end) ===")
    print(
        "Layers: CLI adapters | session | skills | MCP | scheduler tools"
        + (" | SchedulerRunner" if runner else "")
        + "\n"
    )

    try:
        await runtime.run_loop(
            agent_id,
            request=first_request,
            input_source=CLIInput(),
            output_source=CLIOutput(),
            load_last_session=load_last_session,
            last_session_messages=last_session_messages,
        )
    except Exception:
        logger.exception("Failed to run interactive loop")
    finally:
        if runner is not None:
            runner.stop()
        if mcp_clients is not None:
            await mcp_clients.disconnect()
        AgentRuntime._instance = None


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full demo: run_loop + skills + MCP + scheduler + CLI I/O",
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable long memory on agent init (default: on); use --no-use-long-memory to disable",
    )
    parser.add_argument(
        "--user-id",
        default="demo_user",
        help="agent.user_id for session / conversation / long memory",
    )
    parser.add_argument(
        "--no-scheduler",
        action="store_true",
        help="Disable background SchedulerRunner (scheduler CRUD tools still available)",
    )
    parser.add_argument(
        "--scheduler-interval",
        type=float,
        default=5.0,
        help="SchedulerRunner scan interval in seconds (default: 5)",
    )
    args = parser.parse_args()

    try:
        await run_interactive_loop(
            first_request=args.request,
            load_last_session=args.load_last_session,
            last_session_messages=args.last_session_messages,
            use_long_memory=args.use_long_memory,
            user_id=args.user_id or None,
            scheduler_enabled=not args.no_scheduler,
            scheduler_interval=args.scheduler_interval,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
