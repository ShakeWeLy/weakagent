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
    )
    return Message.assistant_message(content)