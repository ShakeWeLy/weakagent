from typing import List, Optional

from weakagent.llm.llm import LLM
from weakagent.memory.message_store import flatten_messages_for_summary
from weakagent.prompt.summary import (
    load_short_memory_summary_system_prompt,
    load_working_memory_summary_system_prompt,
    load_long_memory_summary_system_prompt,
)
from weakagent.schemas.message import Message


SESSION_TITLE_SYSTEM_PROMPT = """
You generate a short conversation session title from the user's first message.

Rules:
- Return ONLY the title text (no quotes, no prefix like "Title:").
- At most 30 characters; prefer concise phrasing in the user's language.
- Capture the main topic or intent, not a full sentence.
"""

SESSION_TITLE_USER_PROMPT = """
User's first message:
{request}

Session title:
"""


LONG_MEMORY_USER_PROMPT = """
User message:
{user_message}
"""

async def summarize_working_memory(llm: LLM, working_memory: List[Message]) -> Message:
    """Extract reusable skills/workflows from working-memory messages."""
    content = await llm.ask(
        messages=flatten_messages_for_summary(working_memory),
        system_msgs=[Message.system_message(load_working_memory_summary_system_prompt())],
        stream=False,
        verbose=True,
    )
    return Message.assistant_message(content or "")


async def summarize_short_memory(llm: LLM, short_memory: List[Message]) -> Message:
    """Compress short-memory messages into a structured conversation summary."""
    if not short_memory:
        return Message.assistant_message("")
    content = await llm.ask(
        messages=flatten_messages_for_summary(short_memory),
        system_msgs=[Message.system_message(load_short_memory_summary_system_prompt())],
        stream=False,
        verbose=True,
    )
    return Message.assistant_message(content or "")


def normalize_long_memory_result(data: Optional[dict]) -> dict:
    """Normalize LLM long-memory extraction output."""
    if not data or not data.get("should_save"):
        return {"should_save": False}

    memory = (data.get("memory") or "").strip()
    if not memory:
        return {"should_save": False}

    importance = data.get("importance", 0.5)
    try:
        importance = float(importance)
    except (TypeError, ValueError):
        importance = 0.5
    importance = max(0.0, min(1.0, importance))

    return {
        "should_save": True,
        "memory_type": str(data.get("memory_type") or "general"),
        "importance": importance,
        "memory": memory,
    }


async def extract_long_memory(llm: LLM, user_message: str) -> dict:
    """Decide whether to store long-term memory from a user message.

    Returns:
        ``{"should_save": false}`` or
        ``{"should_save": true, "memory_type", "importance", "memory"}``.
    """
    from weakagent.utils.json import parse_llm_json_dict

    text = (user_message or "").strip()
    if not text:
        return {"should_save": False}

    content = await llm.ask(
        [Message.user_message(LONG_MEMORY_USER_PROMPT.format(user_message=text))],
        system_msgs=[Message.system_message(load_long_memory_summary_system_prompt())],
        stream=False,
        verbose=True,
    )
    raw = parse_llm_json_dict(content or "")
    return normalize_long_memory_result(raw)


async def generate_session_title(llm: LLM, request: str) -> str:
    """Generate a short session title from the user's first request."""
    content = await llm.ask(
        [Message.user_message(SESSION_TITLE_USER_PROMPT.format(request=request))],
        system_msgs=[Message.system_message(SESSION_TITLE_SYSTEM_PROMPT)],
        stream=False,
        verbose=True,
    )
    return (content or "").strip()
