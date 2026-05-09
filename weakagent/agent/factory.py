from typing import Any, Dict, Optional, Type

from weakagent.agent.base import BaseAgent
from weakagent.agent.brief_react import BriefReActAgent
from weakagent.agent.chat import ChatAgent
from weakagent.agent.multi_react import BriefReActMultiAgent
from weakagent.agent.toolcall import ToolCallAgent
from weakagent.llm import LLM


class AgentFactory:
    """Factory for creating agents with unified dependency injection."""

    def __init__(self):
        self._registry: Dict[str, Type[BaseAgent]] = {
            "chat": ChatAgent,
            "toolcall": ToolCallAgent,
            "brief_react": BriefReActAgent,
            "reacttoolcall": BriefReActAgent,
            "multi_react": BriefReActMultiAgent,
        }

    @property
    def supported_types(self) -> list[str]:
        """Get all supported agent types."""
        return sorted(self._registry.keys())

    def register(self, agent_type: str, agent_cls: Type[BaseAgent]) -> None:
        """Register or override an agent class for a type key."""
        if not agent_type:
            raise ValueError("agent_type cannot be empty")
        if not isinstance(agent_cls, type) or not issubclass(agent_cls, BaseAgent):
            raise TypeError("agent_cls must be a subclass of BaseAgent")
        self._registry[agent_type] = agent_cls

    def register_many(self, mapping: Dict[str, Type[BaseAgent]]) -> None:
        """Batch register multiple agent classes.

        Args:
            mapping: Dict of {agent_type: agent_class} to register.
        """
        for agent_type, agent_cls in mapping.items():
            self.register(agent_type, agent_cls)

    def unregister(self, agent_type: str) -> bool:
        """Unregister an agent type.

        Returns:
            True if removed, False if not found.
        """
        if agent_type in self._registry:
            del self._registry[agent_type]
            return True
        return False

    def create(
        self,
        agent_type: str,
        *,
        config_name: Optional[str] = None,
        llm: Optional[LLM] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseAgent:
        """Create an agent by type with optional LLM/profile injection."""
        agent_cls = self._registry.get(agent_type)
        if agent_cls is None:
            raise ValueError(
                f"Unsupported agent_type: {agent_type}. "
                f"Supported types: {', '.join(self.supported_types)}"
            )

        payload = dict(kwargs)
        if name is not None:
            payload["name"] = name

        if llm is not None:
            payload["llm"] = llm
        elif config_name:
            payload["llm"] = LLM(config_name=config_name)

        return agent_cls(**payload)

