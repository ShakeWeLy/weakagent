from abc import ABC, abstractmethod
import asyncio

class BaseInputSource(ABC):
    @abstractmethod
    async def read(self) -> str:
        pass

class CLIInput(BaseInputSource):
    async def read(self) -> str:
        # input() is a blocking function, so we need to run it in a separate thread
        return await asyncio.to_thread(input, "You> ")

class VoiceInput(BaseInputSource):
    def __init__(self, asr_service):
        self.asr = asr_service

    async def read(self) -> str:
        raise NotImplementedError("VoiceInput is not implemented yet")

class APIInput(BaseInputSource):
    def __init__(self, queue: asyncio.Queue[str]):
        self.queue = queue

    async def read(self) -> str:
        return await self.queue.get()