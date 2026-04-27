from typing import List
from pydantic import BaseModel, Field
from weakagent.schemas.message import Message
from weakagent.schemas.tool import ToolCall
from weakagent.utils.logger import get_logger

logger = get_logger(__name__)


class ShortMemory(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    max_messages: int = Field(default=100)

    def add_message(self, message: Message) -> None:
        """Add a message to memory"""
        self.messages.append(message)
        # Optional: Implement message limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def add_messages(self, messages: List[Message]) -> None:
        """Add multiple messages to memory"""
        self.messages.extend(messages)
        # Optional: Implement message limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]
    
    def add_messages_from_tool_calls(self, tool_calls: List[ToolCall]) -> None:
        """Add messages from tool calls"""
        for tool_call in tool_calls:
            self.add_message(Message.tool_message(
                content=tool_call.function.arguments,
                name=tool_call.function.name,
                tool_call_id=tool_call.id
            ))

    def clear(self) -> None:
        """Clear all messages"""
        self.messages.clear()
    
    def clear_messages(self, n: int) -> None:
        """Clear n messages"""
        self.messages = self.messages[:-n]
        logger.info(f"clear meassage done, now: {self.messages}")


    def get_recent_messages(self, n: int) -> List[Message]:
        """Get n most recent messages"""
        return self.messages[-n:]

    def to_dict_list(self) -> List[dict]:
        """Convert messages to list of dicts"""
        return [msg.to_dict() for msg in self.messages]