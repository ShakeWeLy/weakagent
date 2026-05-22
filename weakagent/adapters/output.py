from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod


class BaseOutputSource(ABC):
    @abstractmethod
    def dispatch(self, message: str) -> None:
        pass

class CLIOutput(BaseOutputSource):
    def dispatch(self, message: str) -> None:
        print(message)

class VoiceOutput(BaseOutputSource):
    def dispatch(self, message: str) -> None:
        raise NotImplementedError("VoiceOutput is not implemented yet")

class APIOutput(BaseOutputSource):
    def __init__(self, queue: asyncio.Queue[str]):
        self.queue = queue

    def dispatch(self, message: str) -> None:
        self.queue.put_nowait(message)