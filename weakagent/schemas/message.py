from enum import Enum
from typing import Any, List, Literal, Optional, Union
from pydantic import BaseModel, Field
from weakagent.schemas.tool import ToolCall
from weakagent.utils.logger import get_logger

class Role(str, Enum):
    """Message role options"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


ROLE_VALUES = tuple(role.value for role in Role)
ROLE_TYPE = Literal[ROLE_VALUES]  # type: ignore

logger = get_logger(__name__)

class Message(BaseModel):
    """Represents a chat message in the conversation"""

    role: ROLE_TYPE = Field(...)  # type: ignore
    content: Optional[str] = Field(default=None)
    tool_calls: Optional[List[ToolCall]] = Field(default=None)
    name: Optional[str] = Field(default=None)
    tool_call_id: Optional[str] = Field(default=None)
    base64_image: Optional[str] = Field(default=None) # base64 encoded image

    def __add__(self, other) -> List["Message"]:
        """支持 Message + list 或 Message + Message 的操作"""
        if isinstance(other, list):
            return [self] + other
        elif isinstance(other, Message):
            return [self, other]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __radd__(self, other) -> List["Message"]:
        """支持 list + Message 的操作"""
        if isinstance(other, list):
            return other + [self]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(other).__name__}' and '{type(self).__name__}'"
            )

    def to_dict(self) -> dict:
        """Convert message to dictionary format"""
        message = {"role": self.role}
        if self.content is not None:
            message["content"] = self.content
        if self.tool_calls is not None:
            message["tool_calls"] = [tool_call.dict() for tool_call in self.tool_calls]
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.base64_image is not None:
            message["base64_image"] = self.base64_image
        return message

    def to_dict_list(self) -> List[dict]:
        """Convert message to list of dicts"""
        return [self.to_dict()]

    def with_truncated_content_middle(
        self,
        *,
        max_chars: int = 800,
        head_ratio: float = 0.7,
        tail_ratio: float = 0.2,
    ) -> "Message":
        """
        Return a copy of this message with `content` truncated in the middle.

        - Only truncates when content is a non-empty string longer than `max_chars`.
        - Keeps the head and tail to preserve context, replaces the middle with an omission marker.
        """
        content = (self.content or "").strip()
        if not content or max_chars <= 0 or len(content) <= max_chars:
            return self

        head_chars = max(1, int(max_chars * head_ratio))
        tail_chars = max(1, int(max_chars * tail_ratio))
        # Ensure we don't exceed max_chars budget and keep at least 1 char in each part.
        if head_chars + tail_chars >= max_chars:
            head_chars = max(1, max_chars // 2)
            tail_chars = max(1, max_chars - head_chars)

        head = content[:head_chars]
        tail = content[-tail_chars:] if tail_chars > 0 else ""
        omitted = len(content) - len(head) - len(tail)
        new_content = f"{head}\n...(omitted {omitted} characters)...\n{tail}"
        return self.model_copy(update={"content": new_content}) # type: ignore

    @classmethod
    def user_message(cls, content: str, base64_image: Optional[str] = None) -> "Message":
        """Create a user message"""
        return cls(role=Role.USER, content=content, base64_image=base64_image)

    @classmethod
    def system_message(cls, content: str) -> "Message":
        """Create a system message"""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def assistant_message(cls, content: Optional[str] = None, base64_image: Optional[str] = None) -> "Message":
        """Create an assistant message"""
        return cls(role=Role.ASSISTANT, content=content, base64_image=base64_image)

    @classmethod
    def tool_message(cls, content: str, name, tool_call_id: str, base64_image: Optional[str] = None) -> "Message":
        """Create a tool message"""
        return cls(
            role=Role.TOOL,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
            base64_image=base64_image,
        )

    @classmethod
    def from_tool_calls(
        cls,
        tool_calls: List[Any],
        content: Union[str, List[str]] = "",
        base64_image: Optional[str] = None,
        **kwargs,
    ) -> "Message":
        """Create ToolCallsMessage from raw tool calls.

        Args:
            tool_calls: Raw tool calls from LLM
            content: Optional message content
            base64_image: Optional base64 encoded image
        """
        def _format_function_obj(func_obj: Any) -> dict:
            """
            Tool call function payload formatter.
            Supports:
            - Pydantic/OpenAI objects with .model_dump()
            - dicts
            - SimpleNamespace / objects with .name + .arguments
            """
            if func_obj is None:
                return {"name": "", "arguments": ""}
            if isinstance(func_obj, dict):
                return func_obj
            model_dump = getattr(func_obj, "model_dump", None)
            if callable(model_dump):
                return model_dump()
            name = getattr(func_obj, "name", "")
            arguments = getattr(func_obj, "arguments", "")
            return {"name": name, "arguments": arguments}

        formatted_calls: List[dict] = []
        for call in tool_calls:
            # Supports OpenAI tool call objects, dicts, or SimpleNamespace-like objects
            if isinstance(call, dict):
                call_id = call.get("id", "")
                call_type = call.get("type", "function")
                func_payload = _format_function_obj(call.get("function"))
            else:
                call_id = getattr(call, "id", "")
                call_type = getattr(call, "type", "function") or "function"
                func_payload = _format_function_obj(getattr(call, "function", None))
            formatted_calls.append({"id": call_id, "function": func_payload, "type": call_type})
        return cls(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=formatted_calls,
            base64_image=base64_image,
            **kwargs,
        )

    def to_truncated_log(self, max_chars: int = 800) -> str:
        """Convert message to truncated log string"""
        content = (self.content or "").strip()
        if len(content) <= max_chars:
            return content

        head = content[:int(max_chars * 0.7)]
        tail = content[-int(max_chars * 0.2)]

        return f"{head}\n...[{len(content)} chars]...\n{tail}"