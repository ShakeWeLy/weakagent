from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type

from weakagent.agent.base import BaseAgent
from weakagent.agent.brief_react import BriefReActAgent
from weakagent.agent.chat import ChatAgent
from weakagent.agent.multi_react import BriefReActMultiAgent
from weakagent.agent.toolcall import ToolCallAgent
from weakagent.llm import LLM

@dataclass
class AgentSpec:
    """Specification of one agent type in the factory registry."""

    agent_cls: Type[BaseAgent]
    default_kwargs: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None

    def build_payload(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Merge default kwargs and call-time kwargs (call-time wins)."""
        merged = dict(self.default_kwargs)
        merged.update(overrides)
        return merged


class AgentFactory:
    """Factory for creating agents with unified dependency injection."""

    def __init__(self):
        self._registry: Dict[str, AgentSpec] = {
            "chat": AgentSpec(agent_cls=ChatAgent),
            "toolcall": AgentSpec(agent_cls=ToolCallAgent),
            "brief_react": AgentSpec(agent_cls=BriefReActAgent),
            "reacttoolcall": AgentSpec(agent_cls=BriefReActAgent),
            "multi_react": AgentSpec(agent_cls=BriefReActMultiAgent),
        }

    @property
    def supported_types(self) -> list[str]:
        """Get all supported agent types."""
        return sorted(self._registry.keys())

    def register_spec(self, agent_type: str, spec: AgentSpec) -> None:
        """Register a full AgentSpec for a type key."""
        self._validate_spec(agent_type, spec)
        self._registry[agent_type] = spec

    def register_many(self, mapping: Dict[str, AgentSpec]) -> None:
        """Batch register multiple AgentSpec items.

        Args:
            mapping: Dict of {agent_type: AgentSpec} to register.
        """
        for agent_type, spec in mapping.items():
            self.register_spec(agent_type, spec)

    def unregister(self, agent_type: str) -> bool:
        """Unregister an agent type.

        Returns:
            True if removed, False if not found.
        """
        if agent_type in self._registry:
            del self._registry[agent_type]
            return True
        return False

    # find spec by agent_type and build payload from kwargs
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
        spec = self._registry.get(agent_type)
        if spec is None:
            raise ValueError(
                f"Unsupported agent_type: {agent_type}. "
                f"Supported types: {', '.join(self.supported_types)}"
            )

        # build payload from kwargs
        payload = spec.build_payload(dict(kwargs))
        # override name and llm if provided
        if name is not None:
            payload["name"] = name
        if llm is not None:
            payload["llm"] = llm
        elif config_name:
            payload["llm"] = LLM(config_name=config_name)

        return spec.agent_cls(**payload)

    @staticmethod
    def _validate_spec(agent_type: str, spec: AgentSpec) -> None:
        if not agent_type:
            raise ValueError("agent_type cannot be empty")
        if not isinstance(spec, AgentSpec):
            raise TypeError("spec must be an AgentSpec")
        if not isinstance(spec.agent_cls, type) or not issubclass(
            spec.agent_cls, BaseAgent
        ):
            raise TypeError("spec.agent_cls must be a subclass of BaseAgent")

