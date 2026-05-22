from weakagent.agent.base import BaseAgent
from weakagent.schemas.agent import AgentState

class EchoAgent(BaseAgent):
    """Minimal agent that echoes the latest user message."""

    async def step(self) -> str:
        latest_user_message = ""
        for msg in reversed(self.messages):
            if msg.role == "user" and msg.content:
                latest_user_message = msg.content
                break
        self.state = AgentState.FINISHED
        return f"{self.name} <= {latest_user_message or '(empty)'}"
