import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from weakagent.llm.summarize import summarize_working_memory
from weakagent.memory.working import WorkingMemory
from weakagent.llm.llm import LLM
from weakagent.schemas.message import Message


working_memory = [

    Message.user_message("Hello, how are you?"),
    Message.assistant_message("I'm good, thank you."),
    Message.user_message("What is your name?"),
    Message.assistant_message("My name is John."),
    Message.user_message("What is location of Tokyo?"),
    Message.assistant_message("Tokyo is in Japan."),
]

llm = LLM()
working_memory = WorkingMemory(messages=working_memory)
summary = asyncio.run(summarize_working_memory(llm, working_memory.messages))
print(summary.content)


