"""WeakAgent: A framework for building agents using LLMs."""

__version__ = "0.1.0"

from .llm import LLM, LLMFactory, TokenCounter

__all__ = ["LLM", "LLMFactory", "TokenCounter", "__version__"]
