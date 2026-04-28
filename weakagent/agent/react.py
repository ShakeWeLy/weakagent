from abc import ABC, abstractmethod
from typing import Optional

from pydantic import Field

from weakagent.agent.base import BaseAgent
from weakagent.llm import LLM
from weakagent.utils.logger import get_logger
from weakagent.schemas.agent import AgentState
from weakagent.memory.short import ShortMemory

logger = get_logger(__name__)


class ReActAgent(BaseAgent, ABC):
    name: str
    description: Optional[str] = None

    system_prompt: Optional[str] = None
    next_step_prompt: Optional[str] = None

    llm: Optional[LLM] = Field(default_factory=LLM)
    memory: ShortMemory = Field(default_factory=ShortMemory)
    state: AgentState = Field(default=AgentState.IDLE, description="Current agent state")

    max_steps: int = Field(default=10, description="Maximum steps before termination")
    current_step: int = Field(default=0, description="Current step in execution")

    @abstractmethod
    async def think(self) -> bool:
        """Process current state and decide next action"""

    @abstractmethod
    async def act(self) -> str:
        """Execute decided actions"""

    async def step(self) -> str:
        """Execute a single step: think and act."""
        should_act = await self.think()
        if not should_act:
            return "Thinking complete - no action needed"
        return await self.act()
