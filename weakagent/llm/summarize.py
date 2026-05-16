from weakagent.llm.llm import LLM
from weakagent.schemas.message import Message
from typing import List

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

async def summarize_short_memory(llm: LLM, short_memory: List[Message]) -> str:
    """Summarize the short memory"""
    content = await llm.ask(
        messages=short_memory,
        system_msgs=[Message.system_message(SHORT_MEMORY_SUMMARY_SYSTEM_PROMPT)],
        stream=True,
    )
    return content