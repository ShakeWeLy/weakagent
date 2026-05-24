import sqlite3
import tomllib
from abc import ABC
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field, model_validator

from weakagent.config.settings import PROJECT_ROOT
from weakagent.schemas.message import Message
from weakagent.schemas.tool import ToolCall
from weakagent.utils.logger import get_logger

logger = get_logger(__name__)

class MemoryType(str, Enum):
    CONVERSATION = "conversation"  # append-only per-message log
    SESSION = "session"  # runtime loop metadata + end-of-runtime summary
    SHORT = "short"  # agent run context 
    WORKING = "working"  # single run context. not include history 
    LONG = "long"  # used after session/runtime complete, to store the long-term memory about user profile and context


class BaseMemory(BaseModel, ABC):
    # common
    messages: List[Message] = Field(default_factory=list)
    max_messages: int = Field(default=100)
    memory_type: MemoryType = Field(..., description="The type of memory")

    # database
    db_path: str = Field(default="weakagent.sqlite3")

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

    # database
    @model_validator(mode="after")
    def _init(self) -> "BaseMemory":
        self.db_path = str(self._resolve_db_path(self.db_path, self.memory_type))
        self._init_db()
        return self

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _resolve_db_path(db_path: str, memory_type: MemoryType) -> Path:
        p = Path(db_path)
        if p.is_absolute():
            return p
        cfg_path = PROJECT_ROOT / "config.toml"
        try:
            raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            raw = {}
        configured = (raw.get(memory_type.value) or {}).get("db_path")
        if configured:
            cp = Path(str(configured))
            return cp if cp.is_absolute() else (PROJECT_ROOT / cp)
        return PROJECT_ROOT / p
    
    def _init_db(self) -> None:
        """Override in subclasses that own sqlite tables."""
        return None

    @staticmethod
    def utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def ensure_db_parent(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)