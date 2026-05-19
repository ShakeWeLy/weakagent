import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from weakagent.agent.base import BaseAgent
from weakagent.agent.factory import AgentFactory
from weakagent.utils.logger import logger

@dataclass
class ManagedAgent:
    """Runtime metadata for a managed agent."""

    agent_id: str
    agent: BaseAgent
    agent_type: str
    parent_id: Optional[str] = None
    children: set[str] = field(default_factory=set)
    task: Optional[asyncio.Task] = None


class AgentRuntime:
    """Manage agent runtime lifecycle, registry, and parent/child relationships.

    This is a singleton - use AgentRuntime.instance() to get the global instance.
    """

    _instance: Optional["AgentRuntime"] = None
    _lock = asyncio.Lock()

    def __init__(self, factory: Optional[AgentFactory] = None):
        if AgentRuntime._instance is not None:
            raise RuntimeError(
                "AgentRuntime is a singleton. Use AgentRuntime.instance() instead."
            )
        AgentRuntime._instance = self
        self.factory = factory or AgentFactory()
        self._agents: Dict[str, ManagedAgent] = {}

    @classmethod
    async def instance(cls, factory: Optional[AgentFactory] = None) -> "AgentRuntime":
        """Get or create the global singleton instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(factory)
        return cls._instance

    @classmethod
    def get_instance(cls) -> Optional["AgentRuntime"]:
        """Get the existing singleton instance (may be None if not initialized)."""
        return cls._instance

    @staticmethod
    def _new_agent_id() -> str:
        return f"agent_{uuid.uuid4().hex[:12]}"

    def register(
        self,
        agent: BaseAgent,
        *,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        """Register an already-created agent and return its managed ID."""
        if parent_id is not None and parent_id not in self._agents:
            raise ValueError(f"parent_id not found: {parent_id}")

        resolved_id = agent_id or self._new_agent_id()
        if resolved_id in self._agents:
            raise ValueError(f"agent_id already exists: {resolved_id}")

        resolved_type = agent_type or agent.__class__.__name__.lower()
        self._agents[resolved_id] = ManagedAgent(
            agent_id=resolved_id,
            agent=agent,
            agent_type=resolved_type,
            parent_id=parent_id,
        )
        # Auto-wire runtime context for multi-agent capable implementations.
        if hasattr(agent, "agent_runtime"):
            setattr(agent, "agent_runtime", AgentRuntime._instance)
        if hasattr(agent, "managed_agent_id"):
            setattr(agent, "managed_agent_id", resolved_id)

        if parent_id:
            self._agents[parent_id].children.add(resolved_id)

        return resolved_id

    def create_agent(
        self,
        agent_type: str,
        *,
        agent_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Create and register an agent via AgentFactory."""
        agent = self.factory.create(agent_type, **kwargs)
        return self.register(
            agent,
            agent_id=agent_id,
            agent_type=agent_type,
            parent_id=parent_id,
        )

    def spawn_sub_agent(self, parent_id: str, agent_type: str, **kwargs: Any) -> str:
        """Create a child agent under a parent agent."""
        if parent_id not in self._agents:
            raise ValueError(f"parent_id not found: {parent_id}")
        return self.create_agent(agent_type, parent_id=parent_id, **kwargs)

    def get(self, agent_id: str) -> BaseAgent:
        """Get managed agent instance by ID."""
        if agent_id not in self._agents:
            raise KeyError(f"agent_id not found: {agent_id}")
        return self._agents[agent_id].agent

    def get_meta(self, agent_id: str) -> ManagedAgent:
        """Get managed metadata by ID."""
        if agent_id not in self._agents:
            raise KeyError(f"agent_id not found: {agent_id}")
        return self._agents[agent_id]

    def list_agents(self, parent_id: Optional[str] = None) -> list[str]:
        """List agent IDs, optionally filtered by parent."""
        if parent_id is None:
            return list(self._agents.keys())
        return [
            agent_id
            for agent_id, meta in self._agents.items()
            if meta.parent_id == parent_id
        ]

    def get_registered_agents(
        self, parent_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return detailed info for registered agents.

        Args:
            parent_id: Optional parent agent ID to filter by.

        Returns:
            List of dicts containing agent details:
            - agent_id: str
            - name: str (agent's name attribute)
            - agent_type: str
            - state: str (current agent state)
            - parent_id: Optional[str]
            - children: list[str] (child agent IDs)
            - current_step: int (current step in execution)
            - max_steps: int (maximum steps allowed)
            - is_running: bool (whether agent has an active background task)
        """
        results: List[Dict[str, Any]] = []

        for agent_id, meta in self._agents.items():
            if parent_id is not None and meta.parent_id != parent_id:
                continue

            agent = meta.agent
            info: Dict[str, Any] = {
                "agent_id": agent_id,
                "name": getattr(agent, "name", "unknown"),
                "agent_type": meta.agent_type,
                "state": getattr(agent, "state", "unknown"),
                "parent_id": meta.parent_id,
                "children": list(meta.children),
                "current_step": getattr(agent, "current_step", 0),
                "max_steps": getattr(agent, "max_steps", 0),
                "is_running": meta.task is not None and not meta.task.done(),
            }
            results.append(info)

        return results

    def get_agent_summaries(self) -> Dict[str, Dict[str, str]]:
        """Return simplified name+description mapping for registered agents."""
        return {
            agent_id: {
                "name": getattr(meta.agent, "name", "unknown"),
                "description": getattr(meta.agent, "description", ""),
            }
            for agent_id, meta in self._agents.items()
        }

    async def run(self, agent_id: str, request: Optional[str] = None) -> str:
        """Run an agent synchronously (await until completion)."""
        agent = self.get(agent_id)
        return await agent.run(request=request)

    async def run_loop(self, agent_id: str, request: Optional[str] = None):
        """Run an agent loop synchronously (await until completion)."""
        try:
            while True:
                request = input("You> ")
                logger.info(f"User request: {request}")
                if not request:
                    continue
                if request.lower() in {"exit", "quit", "q"}:
                    break
                result = await self.run(agent_id, request=request)
                print("\nAgent result:\n", result)
        finally:
            await self.cleanup(agent_id)
            print("Cleanup complete.")
    
    def run_in_background(
        self, agent_id: str, request: Optional[str] = None
    ) -> asyncio.Task:
        """Run an agent in background and return the task."""
        meta = self.get_meta(agent_id)
        if meta.task and not meta.task.done():
            raise RuntimeError(f"agent already running: {agent_id}")

        task = asyncio.create_task(meta.agent.run(request=request))
        meta.task = task
        return task

    async def cancel(self, agent_id: str) -> bool:
        """Cancel background task of an agent if running."""
        meta = self.get_meta(agent_id)
        if not meta.task or meta.task.done():
            return False

        meta.task.cancel()
        try:
            await meta.task
        except asyncio.CancelledError:
            logger.info(f"Cancelled agent task: {agent_id}")
        except Exception as exc:
            logger.warning(f"Agent task ended with error after cancel: {exc}")
        return True

    async def cleanup(self, agent_id: str, *, recursive: bool = True) -> None:
        """Cancel tasks, cleanup resources, and remove agent from registry."""
        meta = self.get_meta(agent_id)

        child_ids = list(meta.children) if recursive else []
        for child_id in child_ids:
            if child_id in self._agents:
                await self.cleanup(child_id, recursive=True)

        await self.cancel(agent_id)

        cleanup = getattr(meta.agent, "cleanup", None)
        if callable(cleanup):
            maybe_coro = cleanup()
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro

        if meta.parent_id and meta.parent_id in self._agents:
            self._agents[meta.parent_id].children.discard(agent_id)

        self._agents.pop(agent_id, None)

    async def cleanup_all(self) -> None:
        """Cleanup all managed agents."""
        for agent_id in list(self._agents.keys()):
            if agent_id in self._agents:
                await self.cleanup(agent_id, recursive=True)
