from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from weakagent.memory.message import Message


def to_extra_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        data = dump()
        return data if isinstance(data, dict) else {}
    return {}


def tool_call_id(tc: Any) -> Optional[str]:
    if isinstance(tc, dict):
        raw = tc.get("id")
        return str(raw) if raw is not None else None
    raw = getattr(tc, "id", None)
    return str(raw) if raw is not None else None


def split_system_prefix(msgs: List[Message]) -> tuple[List[Message], List[Message]]:
    sys_prefix: List[Message] = []
    rest: List[Message] = []
    for m in msgs:
        if not rest and m.role == "system":
            sys_prefix.append(m)
        else:
            rest.append(m)
    return sys_prefix, rest


def find_assistant_for_tool(messages: List[Message], tool_idx: int) -> Optional[int]:
    tool_msg = messages[tool_idx]
    tcid = tool_msg.tool_call_id
    if not tcid:
        return None
    for j in range(tool_idx - 1, -1, -1):
        prev = messages[j]
        if prev.role == "assistant" and prev.tool_calls:
            ids = {tool_call_id(tc) for tc in prev.tool_calls}
            ids.discard(None)
            if tcid in ids:
                return j
        if prev.role == "user":
            break
    return None


def expand_start_for_tool_integrity(messages: List[Message], start: int) -> int:
    start = max(0, start)
    while True:
        expanded = start
        for i in range(start, len(messages)):
            if messages[i].role != "tool":
                continue
            parent = find_assistant_for_tool(messages, i)
            if parent is not None and parent < expanded:
                expanded = parent
        if expanded == start:
            return start
        start = expanded


def select_last_n_messages_with_integrity(messages: List[Message], n: int) -> List[Message]:
    """Take the last n non-system messages while preserving tool-call chains."""
    n = max(1, int(n))
    sys_prefix, rest = split_system_prefix(messages)
    if not rest:
        return list(sys_prefix)
    if n >= len(rest):
        return sys_prefix + rest
    start = expand_start_for_tool_integrity(rest, len(rest) - n)
    return sys_prefix + rest[start:]


def _tool_call_names(tool_calls: Any) -> List[str]:
    names: List[str] = []
    if not tool_calls:
        return names
    for tc in tool_calls:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
        else:
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", None) if fn is not None else None
        if name:
            names.append(str(name))
    return names


def flatten_messages_for_summary(messages: List[Message]) -> List[Message]:
    """Plain chat transcript for summarization (no thinking/tool-call replay)."""
    flat: List[Message] = []
    for m in messages:
        if m.role == "system":
            flat.append(
                m.model_copy(update={"reasoning_content": None, "tool_calls": None})
            )
        elif m.role == "user":
            flat.append(
                m.model_copy(update={"reasoning_content": None, "tool_calls": None})
            )
        elif m.role == "assistant":
            text = (m.content or "").strip()
            if not text and m.tool_calls:
                names = _tool_call_names(m.tool_calls)
                if names:
                    text = f"[tools: {', '.join(names)}]"
            if text:
                flat.append(Message.assistant_message(text))
        elif m.role == "tool":
            label = m.name or "tool"
            flat.append(Message.user_message(f"[{label}]: {m.content or ''}"))
    return flat


def message_from_storage_row(row: Any) -> Message:
    """Rebuild a Message from a persisted message row (session/runtime)."""
    role = str(row["role"])
    content = row["content"]
    try:
        extra = json.loads(row["extra"] or "{}")
    except Exception:
        extra = {}

    tool_calls_raw = extra.get("tool_calls")
    if role == "assistant" and tool_calls_raw:
        return Message.from_tool_calls(
            tool_calls_raw,
            content=content or "",
            reasoning_content=extra.get("reasoning_content"),
        )

    return Message(
        role=role,  # type: ignore[arg-type]
        content=content,
        name=extra.get("name"),
        tool_call_id=extra.get("tool_call_id"),
        reasoning_content=extra.get("reasoning_content"),
    )
