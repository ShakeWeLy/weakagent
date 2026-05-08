from weakagent.llm.llm import LLM
from weakagent.schemas.message import Message
from weakagent.agent.base import BaseAgent
from weakagent.memory.short import ShortMemory
from weakagent.schemas.agent import AgentState

class ChatAgent(BaseAgent):
    name: str = "chat"
    description: str = "A chat agent that can chat with the user."
    system_prompt: str = "You are a chat agent that can chat with the user."
    next_step_prompt: str = "You are a chat agent that can chat with the user."
    llm: LLM = LLM()
    memory: ShortMemory = ShortMemory()
    state: AgentState = AgentState.IDLE
    max_steps: int = 1
    current_step: int = 0
    duplicate_threshold: int = 2

    async def step(self) -> str:
        """Execute a single step in the agent's workflow."""
        content = await self.llm.ask(self.memory.messages, system_msgs=[Message.system_message(self.system_prompt)], temperature=0.0, verbose=True)
        self.memory.add_message(Message.assistant_message(content))
        return content