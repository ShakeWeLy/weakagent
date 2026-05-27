from __future__ import annotations

from typing import Any, Dict, List

from weakagent.tools.base import BaseTool, ToolExecutionResult


class CreateSubAgentTool(BaseTool):
    name: str = "create_sub_agent"
    description: str = (
        "Create a sub-agent without running it. "
        "Supports custom init fields such as name, system_prompt, max_steps, and tools."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "sub_agent_name": {
                "type": "string",
                "description": "Sub-agent factory name to create.",
            },
            "caller_agent_id": {
                "type": "string",
                "description": "Optional parent/caller managed agent id.",
            },
            "name": {
                "type": "string",
                "description": "Optional sub-agent instance name.",
            },
            "system_prompt": {
                "type": "string",
                "description": "Optional system prompt for the sub-agent.",
            },
            "next_step_prompt": {
                "type": "string",
                "description": "Optional next-step prompt for the sub-agent.",
            },
            "max_steps": {
                "type": "integer",
                "description": "Optional max step limit for the sub-agent.",
            },
            "tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional tool names for available_tools injection. "
                    "Supported: create_chat_completion, terminate, summary, "
                    "run_sub_agent, create_sub_agent, hot_reload."
                ),
            },
            "extra_kwargs": {
                "type": "object",
                "description": "Optional extra kwargs passed to AgentFactory.create().",
            },
        },
        "required": ["sub_agent_name"],
    }

    async def execute(
        self,
        sub_agent_name: str,
        caller_agent_id: str | None = None,
        name: str | None = None,
        system_prompt: str | None = None,
        next_step_prompt: str | None = None,
        max_steps: int | None = None,
        tools: List[str] | None = None,
        extra_kwargs: Dict[str, Any] | None = None,
    ) -> ToolExecutionResult:
        from weakagent.agent.runtime import AgentRuntime

        if not sub_agent_name:
            return self.fail_response("sub_agent_name is required")

        runtime = AgentRuntime.get_instance()
        if runtime is None:
            return self.fail_response(
                "AgentRuntime singleton not initialized. "
                "Call `await AgentRuntime.instance()` first."
            )

        if sub_agent_name not in runtime.factory.supported_types:
            return self.fail_response(
                f"Unknown sub-agent name: {sub_agent_name}. "
                f"Supported types: {', '.join(runtime.factory.supported_types)}"
            )

        payload: Dict[str, Any] = dict(extra_kwargs or {})
        if name is not None:
            payload["name"] = name
        if system_prompt is not None:
            payload["system_prompt"] = system_prompt
        if next_step_prompt is not None:
            payload["next_step_prompt"] = next_step_prompt
        if max_steps is not None:
            payload["max_steps"] = max_steps

        if tools is not None:
            try:
                payload["available_tools"] = self._build_tool_collection(tools)
            except ValueError as exc:
                return self.fail_response(str(exc))

        try:
            sub_agent_id = runtime.create_agent(
                sub_agent_name,
                parent_id=caller_agent_id,
                **payload,
            )
        except Exception as exc:
            return self.fail_response(
                f"Failed to create sub-agent `{sub_agent_name}`: {exc}"
            )

        if caller_agent_id:
            try:
                caller = runtime.get(caller_agent_id)
                if hasattr(caller, "current_sub_agent_id"):
                    setattr(caller, "current_sub_agent_id", sub_agent_id)
            except Exception:
                pass

        return self.success_response(
            {
                "sub_agent_name": sub_agent_name,
                "sub_agent_id": sub_agent_id,
                "caller_agent_id": caller_agent_id,
                "config": {
                    "name": name,
                    "system_prompt": system_prompt,
                    "next_step_prompt": next_step_prompt,
                    "max_steps": max_steps,
                    "tools": tools,
                    "extra_kwargs": extra_kwargs or {},
                },
                "message": f"Sub-agent `{sub_agent_name}` created successfully",
            }
        )

    @staticmethod
    def _build_tool_collection(tool_names: List[str]):
        from weakagent.tools.create_chat_completion import CreateChatCompletion
        from weakagent.tools import Terminate
        from weakagent.tools.tool_collection import ToolCollection
        from weakagent.tools.sub_agent.run_sub_agent import RunSubAgentTool
        from weakagent.tools.tool.hot_reload import HotReloadTool
        from weakagent.tools.memory.long import SaveLongMemoryTool

        builders = {
            "create_chat_completion": CreateChatCompletion,
            "terminate": Terminate,
            "run_sub_agent": RunSubAgentTool,
            "create_sub_agent": CreateSubAgentTool,
            "save_long_memory": SaveLongMemoryTool,
            "hot_reload": HotReloadTool,
        }
        unsupported = sorted({n for n in tool_names if n not in builders})
        if unsupported:
            raise ValueError(
                f"Unsupported tool names: {', '.join(unsupported)}. "
                f"Supported: {', '.join(sorted(builders.keys()))}"
            )

        return ToolCollection(*[builders[name]() for name in tool_names])
