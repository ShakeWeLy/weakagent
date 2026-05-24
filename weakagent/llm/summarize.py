from weakagent.llm.llm import LLM
from weakagent.schemas.message import Message
from typing import List, Optional
from weakagent.prompt.summary import load_short_memory_summary_system_prompt, load_working_memory_summary_system_prompt


async def summarize_working_memory(llm: LLM, working_memory: List[Message]) -> Message:
    """Extract reusable skills/workflows from working-memory messages."""
    content = await llm.ask(
        messages=working_memory,
        system_msgs=[Message.system_message(load_working_memory_summary_system_prompt())],
        stream=False,
    )
    return Message.assistant_message(content or "")


async def summarize_short_memory(llm: LLM, short_memory: List[Message]) -> Message:
    """Compress short-memory messages into a structured conversation summary."""
    if not short_memory:
        return Message.assistant_message("")
    content = await llm.ask(
        messages=short_memory,
        system_msgs=[Message.system_message(load_short_memory_summary_system_prompt())],
        stream=False,
    )
    return Message.assistant_message(content or "")


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

LONG_MEMORY_SUMMARY_SYSTEM_PROMPT = """
You are a Long-Term Memory Extractor for an AI Agent.

Your task:
Determine whether the user's message contains information worth storing as long-term memory.

Definition of long-term memory:
Information that may still influence the AI assistant's behavior, response style, tool usage, or decision-making weeks or months later.

Information worth saving includes:
- User identity / profession / domain
- Long-term projects
- Technical stack
- User preferences
- Work habits
- Long-term goals
- Important confirmed decisions

Do NOT save:
- Temporary questions
- One-time errors
- Current weather/location
- Casual small talk
- Short-term conversational context

Output JSON only:

{
  "should_save": true,
  "memory_type": "project",
  "importance": 0.82,
  "memory": "User is developing a YOLO pruning system"
}

Requirements:
- Output JSON only
- If not worth saving:
{
  "should_save": false
}
- The memory field must be concise, structured, and deduplicated
- Do not copy the user's full original message
"""

LONG_MEMORY_USER_PROMPT = """
User message:
{user_message}
"""


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
        system_msgs=[Message.system_message(LONG_MEMORY_SUMMARY_SYSTEM_PROMPT)],
        stream=False,
    )
    raw = parse_llm_json_dict(content or "")
    return normalize_long_memory_result(raw)


async def generate_session_title(llm: LLM, request: str) -> str:
    """Generate a short session title from the user's first request."""
    content = await llm.ask(
        [Message.user_message(SESSION_TITLE_USER_PROMPT.format(request=request))],
        system_msgs=[Message.system_message(SESSION_TITLE_SYSTEM_PROMPT)],
        stream=False,
    )
    return (content or "").strip()
