"""Tools module for WeakAgent."""
from .terminate import TerminateTool as Terminate
from .ask_human import AskHumanTool
__all__ = [
    "Terminate",
    "AskHumanTool",
]