"""Meta-tool: reload Python modules and remount tools without process restart."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from weakagent.tools.base import BaseTool, ToolExecutionResult
from weakagent.utils.module_reload import reload_modules, reload_weakagent_tool_modules

if TYPE_CHECKING:
    from weakagent.agent.base import BaseAgent


class HotReloadTool(BaseTool):
    """Reload ``weakagent.*`` modules after code edits (development aid).

    Pitfall: reload only refreshes Python module objects. Already-running asyncio
    tasks, pydantic model classes constructed at import time, and singletons may
    still hold stale references. Remount affected tools on the current agent when
    possible.
    """

    name: str = "hot_reload"
    description: str = (
        "Development meta-tool: reload weakagent Python modules from disk using "
        "importlib.reload, refresh the built-in tool registry, and optionally "
        "remount named tools on the current agent. Use after patch_file/write_file "
        "changes tool or sub-agent code. Example module_names: "
        "['weakagent.tools.memory.long', 'weakagent.tools.sub_agent.create_sub_agent']."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "module_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Dotted module paths under weakagent.* to reload. "
                    "Shorthand like tools.memory.long is accepted."
                ),
            },
            "remount_tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional built-in tool names to re-instantiate on the current "
                    "agent (e.g. save_long_memory, create_sub_agent)."
                ),
            },
            "use_tool_defaults": {
                "type": "boolean",
                "description": (
                    "When true, reload a curated set of tool/sub-agent modules "
                    "before module_names. Default false."
                ),
                "default": False,
            },
        },
        "required": [],
    }

    async def execute(self, **kwargs) -> ToolExecutionResult:
        return self.fail_response(
            "hot_reload must run inside an agent tool loop (use execute_for_agent)."
        )

    async def execute_for_agent(
        self,
        agent: BaseAgent,
        *,
        module_names: List[str],
        remount_tools: Optional[List[str]] = None,
        use_tool_defaults: bool = False,
        **kwargs,
    ) -> ToolExecutionResult:
        if not module_names and not use_tool_defaults:
            return self.fail_response(
                "Provide module_names and/or set use_tool_defaults=true."
            )

        reload_results: List[Dict[str, Any]] = []
        if use_tool_defaults:
            reload_results.extend(reload_weakagent_tool_modules(extra_modules=module_names))
        elif module_names:
            reload_results.extend(reload_modules(module_names))
            from weakagent.tools.tool_collection import get_builtin_tool_registry

            get_builtin_tool_registry(refresh=True)

        remounted: List[str] = []
        remount_errors: List[str] = []
        collection = getattr(agent, "available_tools", None)
        if remount_tools and collection is not None:
            for tool_name in remount_tools:
                name = (tool_name or "").strip()
                if not name:
                    continue
                try:
                    mounted = collection.remount_tool_by_name(name)
                except Exception as exc:
                    remount_errors.append(f"{name}: {exc}")
                    continue
                if mounted is None:
                    remount_errors.append(f"{name}: not found in built-in registry")
                else:
                    remounted.append(mounted.name)

        ok = all(r.get("status") == "reloaded" for r in reload_results)
        payload: Dict[str, Any] = {
            "reloaded_modules": reload_results,
            "remounted_tools": remounted,
            "remount_errors": remount_errors,
        }
        summary_lines = [
            f"Reloaded {sum(1 for r in reload_results if r.get('status') == 'reloaded')} module(s).",
        ]
        if remounted:
            summary_lines.append(f"Remounted: {', '.join(remounted)}.")
        if remount_errors:
            summary_lines.append(f"Remount errors: {'; '.join(remount_errors)}.")
        failed = [r for r in reload_results if r.get("status") != "reloaded"]
        if failed:
            summary_lines.append(
                "Reload errors: "
                + "; ".join(f"{r.get('module')}: {r.get('error')}" for r in failed)
            )

        if not ok or remount_errors:
            payload["message"] = "\n".join(summary_lines)
            return ToolExecutionResult(
                success=False,
                output="\n".join(summary_lines),
                error="; ".join(summary_lines),
                data=payload,
            )

        payload["message"] = "\n".join(summary_lines)
        return self.success_response(payload)
