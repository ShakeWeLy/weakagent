"""Agent module for WeakAgent."""
from .base import BaseAgent
from .brief_react import BriefReActAgent
from .toolcall import ToolCallAgent

__all__ = ["BaseAgent", "BriefReActAgent", "ToolCallAgent"]