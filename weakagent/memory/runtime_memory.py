from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from weakagent.memory.base import BaseMemory, MemoryType
from weakagent.schemas.message import Message


class RuntimeMemory(BaseMemory):
    """Agent-level runtime memory.

    Keeps ONLY:
    - the request input (user)
    - the final result of each run (assistant)

    It is NOT cleared after `agent.run()` finishes.
    """

    messages: List[Message] = Field(default_factory=list)
    memory_type: MemoryType = Field(default=MemoryType.RUNTIME)

    def add_request(self, request: Optional[str]) -> None:
        if request:
            self.add_message(Message.user_message(request))

    def add_last_result(self, last_result: Optional[str]) -> None:
        if last_result is not None:
            self.add_message(Message.assistant_message(last_result))

