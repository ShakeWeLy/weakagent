"""Memory module for WeakAgent."""

import importlib
from typing import Any

from .message import Message
from .base import BaseMemory, MemoryType

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "MemoryCleanupStrategy": (".short", "MemoryCleanupStrategy"),
    "ShortMemory": (".short", "ShortMemory"),
    "ShortMemoryFlushResult": (".short", "ShortMemoryFlushResult"),
    "ShortMemorySnapshotEntry": (".short", "ShortMemorySnapshotEntry"),
    "ConversationMemory": (".conversation", "ConversationMemory"),
    "SessionMemory": (".session", "SessionMemory"),
    "SessionMemorySummaryEntry": (".session", "SessionMemorySummaryEntry"),
    "SessionRecord": (".session", "SessionRecord"),
    "LongMemory": (".long", "LongMemory"),
    "LongMemoryEntry": (".long", "LongMemoryEntry"),
    "WorkingMemory": (".working", "WorkingMemory"),
    "WorkingMemorySummaryEntry": (".working", "WorkingMemorySummaryEntry"),
}

__all__ = [
    "Message",
    "BaseMemory",
    "MemoryCleanupStrategy",
    "MemoryType",
    "ShortMemory",
    "ShortMemoryFlushResult",
    "ShortMemorySnapshotEntry",
    "ConversationMemory",
    "SessionMemory",
    "SessionMemorySummaryEntry",
    "SessionRecord",
    "LongMemory",
    "LongMemoryEntry",
    "WorkingMemory",
    "WorkingMemorySummaryEntry",
]


def __getattr__(name: str) -> Any:
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = spec
    module = importlib.import_module(module_name, __name__)
    return getattr(module, attr_name)
