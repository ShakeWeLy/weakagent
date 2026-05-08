from weakagent.llm.llm import LLM
from weakagent.schemas.message import Message
from weakagent.agent.base import BaseAgent
from weakagent.memory.short import ShortMemory
from weakagent.schemas.agent import AgentState
from weakagent.memory.base import MemoryCleanupStrategy

CHAT_SYSTEM_PROMPT = """
You are a helpful and natural conversational assistant.

Behavior rules:
- Answer the user's latest message directly.
- Use conversation history when it is relevant.
- Keep responses clear, concise, and accurate.
- Do not make up facts or pretend to know things you do not know.
- If information is missing or ambiguous, ask a brief clarifying question.
- Maintain conversational continuity across turns.
- Avoid unnecessary repetition.

Context rules:
- The last message with role=user is the current user request.
- Previous messages are conversation history and may provide context.
"""


class ChatAgent(BaseAgent):
    name: str = "chat"
    description: str = "A chat agent that can chat with the user."
    system_prompt: str = CHAT_SYSTEM_PROMPT
    next_step_prompt: str = "You are a chat agent that can chat with the user."
    llm: LLM = LLM()
    memory: ShortMemory = ShortMemory(
        cleanup_strategy=MemoryCleanupStrategy.TRUNCATE_TOOL_OUTPUT,
        truncate_tool_chars=1500,
        keep_last_n=50,
    )

    state: AgentState = AgentState.IDLE
    max_steps: int = 1
    current_step: int = 0
    duplicate_threshold: int = 2

    async def step(self) -> str:
        """Execute a single step in the agent's workflow."""
        content = await self.llm.ask(self.memory.messages, system_msgs=[Message.system_message(self.system_prompt)], temperature=0.0, verbose=True)
        self.memory.add_message(Message.assistant_message(content))
        return content