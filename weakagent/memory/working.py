import asyncio
from typing import List
from pydantic import BaseModel, Field
from weakagent.schemas.message import Message
from weakagent.schemas.tool import ToolCall
from weakagent.utils.logger import get_logger
from weakagent.memory.base import BaseMemory, MemoryType
from weakagent.llm.summarize import summarize_working_memory
from weakagent.llm.llm import LLM

logger = get_logger(__name__)
llm = LLM()

class WorkingMemory(BaseMemory):
    messages: List[Message] = Field(default_factory=list)
    memory_type: MemoryType = Field(default=MemoryType.WORKING)

    def summarize(self) -> Message:
        """Summarize the working memory"""
        summary = asyncio.run(summarize_working_memory(llm, self.messages))
        return summary
