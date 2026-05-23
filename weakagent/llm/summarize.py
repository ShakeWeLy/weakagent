from weakagent.llm.llm import LLM
from weakagent.schemas.message import Message
from typing import List, Optional

WORKING_MEMORY_SUMMARY_SYSTEM_PROMPT = """
You are a helpful assistant that summarizes the working memory, only return the summary, no other text.

[Example]
Working Memory:
- User: Hello, how are you?
- Assistant: I'm good, thank you.
- User: What is your name?
- Assistant: My name is John.
- User: What is your favorite color?
- Assistant: My favorite color is blue.
Summary: 
User asked about the assistant's name and favorite color. Assistant answered the questions.

"""
WORKING_MEMORY_SUMMARY_USER_PROMPT = """
return the summary of the working memory:
{working_memory}
"""


async def summarize_working_memory(llm: LLM, working_memory: List[Message]) -> Message:
    """Summarize the working memory"""
    content = await llm.ask(
        [
            Message.user_message(
                WORKING_MEMORY_SUMMARY_USER_PROMPT.format(working_memory=working_memory)
            )
        ],
        system_msgs=[Message.system_message(WORKING_MEMORY_SUMMARY_SYSTEM_PROMPT)],
        stream=False,
    )
    return Message.assistant_message(content)


SHORT_MEMORY_SUMMARY_SYSTEM_PROMPT = """
You are a conversation summarization module for an AI agent system.

Your task is to compress the given multi-turn conversation history into a concise, structured summary that preserves all important information for downstream reasoning.

## Input
You will receive a conversation history between a user and an AI agent, including:
- user messages
- assistant responses
- tool calls (if any)
- intermediate reasoning or actions (if present)

## Output Requirements
Produce a structured summary that includes the following sections:

### 1. User Intent
Summarize the user's main goals, tasks, or problems.

### 2. Key Information
Extract and list important facts, constraints, preferences, or context provided by the user.

### 3. Actions Taken
Summarize what the assistant or agent has already done (e.g., code generated, tools used, decisions made).

### 4. Current State
Describe the current status of the task:
- what is completed
- what is in progress
- what is unresolved

### 5. Open Questions / Next Steps
List what still needs to be done or clarified.

## Rules
- Be concise but information-dense.
- Do NOT include irrelevant chat content or small talk.
- Do NOT hallucinate new information.
- Preserve technical details (APIs, parameters, errors, file paths, etc.).
- If tool calls exist, summarize their purpose and result, not raw logs.
- If the conversation is short, still follow the structure but keep it minimal.

## Output Format
Return ONLY the structured summary in Markdown format.
No extra explanation or commentary.
"""

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


async def summarize_short_memory(llm: LLM, short_memory: List[Message]) -> str:
    """Summarize the short memory"""
    content = await llm.ask(
        messages=short_memory,
        system_msgs=[Message.system_message(SHORT_MEMORY_SUMMARY_SYSTEM_PROMPT)],
        stream=True,
    )
    return content