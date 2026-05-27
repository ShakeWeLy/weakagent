"""Persist explicit long-term memory for a user (sqlite)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from weakagent.memory.long import LongMemory
from weakagent.tools.base import BaseTool, ToolExecutionResult

if TYPE_CHECKING:
    from weakagent.agent.base import BaseAgent


def _long_memory_for_agent(agent: BaseAgent) -> Tuple[LongMemory, Optional[str]]:
    """Return runtime long-memory store and user_id for the executing agent."""
    user_id = getattr(agent, "user_id", None)
    mem = getattr(agent, "runtime_long_memory", None)
    if not isinstance(mem, LongMemory):
        mem = LongMemory(user_id=user_id)
        agent.runtime_long_memory = mem
    elif user_id:
        mem.user_id = user_id
        if not mem.entries:
            mem.load_for_user(user_id)
    return mem, user_id


def _source_message_from_agent(agent: BaseAgent) -> Optional[str]:
    """Latest user turn from short_memory (fallback: last_request)."""
    for msg in reversed(agent.short_memory.messages):
        if msg.role == "user" and (msg.content or "").strip():
            return (msg.content or "").strip()[:2000]
    last = getattr(agent, "last_request", None)
    return (last or "").strip()[:2000] or None


def _is_long_memory_system_message(content: str) -> bool:
    return (content or "").startswith("[Long-term memory]")


def _refresh_agent_long_memory(agent: BaseAgent, long_mem: LongMemory) -> None:
    """Reload long-memory context into agent.system_messages when enabled."""
    from weakagent.schemas.message import Message

    agent.runtime_long_memory = long_mem
    if not getattr(agent, "use_long_memory", False):
        return

    agent.long_memory = long_mem
    if long_mem.user_id:
        long_mem.load_for_user(long_mem.user_id)
    ctx = long_mem.to_system_context()
    if not ctx:
        agent.system_messages = [
            msg
            for msg in agent.system_messages
            if not _is_long_memory_system_message(msg.content or "")
        ]
        agent.long_memory_message = None
        return

    new_msg = Message.system_message(ctx)
    agent.long_memory_message = new_msg
    for i, msg in enumerate(agent.system_messages):
        if _is_long_memory_system_message(msg.content or ""):
            agent.system_messages[i] = new_msg
            return
    agent.system_messages.append(new_msg)


class SaveLongMemoryTool(BaseTool):
    """Save durable user facts/preferences/projects into long-term memory."""

    name: str = "save_long_memory"
    description: str = (
        "Persist a concise long-term memory about the user (identity, preferences, "
        "projects, stack, goals). Uses the current agent's user_id and the latest user "
        "message from short_memory as context. Skips exact duplicates for the same user."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "memory": {
                "type": "string",
                "description": (
                    "Concise, structured fact to remember (not the user's full message)."
                ),
            },
            "memory_type": {
                "type": "string",
                "description": (
                    "Category, e.g. general, preference, project, identity, goal, habit."
                ),
                "default": "general",
            },
            "importance": {
                "type": "number",
                "description": "Salience from 0.0 to 1.0 (default 0.75).",
                "default": 0.75,
            },
        },
        "required": ["memory"],
    }

    async def execute(
        self,
        memory: str,
        memory_type: str = "general",
        importance: float = 0.75,
    ) -> ToolExecutionResult:
        return self.fail_response(
            "save_long_memory must run inside an agent tool loop "
            "(use execute_for_agent)."
        )

    async def execute_for_agent(
        self,
        agent: BaseAgent,
        *,
        memory: str,
        memory_type: str = "general",
        importance: float = 0.75,
    ) -> ToolExecutionResult:
        text = (memory or "").strip()
        if not text:
            return self.fail_response("`memory` must be a non-empty string.")

        try:
            long_mem, resolved_user_id = _long_memory_for_agent(agent)
        except Exception as exc:
            return self.fail_response(f"Failed to open long-term memory store: {exc}")

        source_message = _source_message_from_agent(agent)
        entry = long_mem.add_entry(
            content=text,
            memory_type=(memory_type or "general").strip() or "general",
            importance=importance,
            user_id=resolved_user_id,
            source_message=source_message,
        )

        if entry is None:
            return self.fail_response(
                "Long-term memory was not saved (empty content or duplicate for this user)."
            )

        _refresh_agent_long_memory(agent, long_mem)

        payload: Dict[str, Any] = {
            "memory_id": entry.memory_id,
            "user_id": entry.user_id,
            "memory_type": entry.memory_type,
            "importance": entry.importance,
            "content": entry.content,
            "source_message": entry.source_message,
            "agent_id": getattr(agent, "managed_agent_id", None),
        }
        payload["message"] = (
            f"Saved long-term memory `{entry.memory_id}` for user_id={entry.user_id!r}."
        )
        return self.success_response(payload)
