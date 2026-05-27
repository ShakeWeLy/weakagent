import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from weakagent.agent.base import BaseAgent
from weakagent.agent.factory import AgentFactory
from weakagent.agent.run_modes import AgentRunMixin
from weakagent.llm.llm import LLM
from weakagent.memory.conversation import ConversationMemory
from weakagent.memory.long import LongMemory
from weakagent.memory.session import SessionMemory
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
    queue_task: Optional[asyncio.Task] = None


class AgentRuntime(AgentRunMixin):
    """Manage agent runtime lifecycle, registry, and parent/child relationships.

    Session transcript is persisted in :class:`ConversationMemory`; metadata and
    end-of-loop summaries in :class:`SessionMemory`; long-term facts in :class:`LongMemory`.

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
        self.request_queue: asyncio.Queue[str] = asyncio.Queue()
        self.result_queue: asyncio.Queue[str] = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @classmethod
    async def instance(cls, factory: Optional[AgentFactory] = None) -> "AgentRuntime":
        """Get or create the global singleton instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(factory)
                    cls._instance._loop = asyncio.get_running_loop()
        return cls._instance

    @classmethod
    def get_instance(cls) -> Optional["AgentRuntime"]:
        """Get the existing singleton instance (may be None if not initialized)."""
        return cls._instance

    @staticmethod
    def _new_agent_id() -> str:
        return f"agent_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _agent_long_memory(agent: BaseAgent) -> LongMemory:
        mem = getattr(agent, "runtime_long_memory", None)
        if not isinstance(mem, LongMemory):
            mem = LongMemory()
            agent.runtime_long_memory = mem
        return mem

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

        self._wire_session(agent, agent_id=resolved_id, agent_type=resolved_type)
        return resolved_id

    def _load_last_runtime_session(
        self,
        agent_id: str,
        *,
        load_last_session: bool,
        last_session_messages: int = 10,
    ) -> None:
        """Hydrate in-memory session + short memory from the prior runtime loop.

        Messages are read from :class:`ConversationMemory` for the latest closed
        session of this ``agent_id`` (not the new empty ``session_id``).
        """
        if not load_last_session:
            return
        agent = self.get(agent_id)
        sess = agent.session
        if sess is None:
            return
        if sess.messages:
            logger.debug(
                "Skip load_last_session: session already has %s message(s)",
                len(sess.messages),
            )
            return
        try:
            prior_session_id = SessionMemory.get_last_session_id_for_agent(
                agent_id,
                db_path=sess.db_path,
                exclude_session_id=sess.session_id,
            )
            last_n = last_session_messages if last_session_messages > 0 else None
            if prior_session_id:
                messages = ConversationMemory.fetch_messages(
                    db_path=sess.db_path,
                    session_id=prior_session_id,
                    agent_id=agent_id,
                    last_n=last_n,
                    exclude_roles=("system",),
                )
            else:
                messages = ConversationMemory.fetch_messages(
                    db_path=sess.db_path,
                    agent_id=agent_id,
                    last_n=last_n,
                    exclude_roles=("system",),
                )
            if not messages:
                return
            sess.messages = list(messages)
            agent.short_memory.messages = list(messages)
            logger.info(
                "Loaded %s message(s) from prior session for agent_id=%s",
                len(messages),
                agent_id,
            )
        except Exception:
            logger.exception(
                "Failed to load last session for agent_id=%s", agent_id
            )

    def _wire_session(self, agent: BaseAgent, *, agent_id: str, agent_type: str) -> None:
        """Bind managed agent metadata to session, conversation, short, and long memory."""
        sess = agent.session
        if sess is None:
            return

        sess.agent_id = agent_id
        sess.agent_type = sess.agent_type or agent_type
        agent.short_memory.session_id = sess.session_id

        conv = agent.conversation
        if conv is not None:
            conv.session_id = sess.session_id
            conv.agent_id = agent_id
            conv.agent_type = conv.agent_type or agent_type
            if sess.user_id:
                conv.user_id = conv.user_id or sess.user_id

        uid = agent.user_id or sess.user_id
        if uid:
            agent.user_id = uid
            sess.user_id = uid
            if conv is not None:
                conv.user_id = uid
            self._agent_long_memory(agent).user_id = uid

        try:
            sess.ensure_session()
        except Exception:
            logger.exception("Failed to ensure session for agent_id=%s", agent_id)

    def _wire_runtime_session(
        self, agent: BaseAgent, *, agent_id: str, agent_type: str
    ) -> None:
        """Alias for ``_wire_session`` (kept for compatibility)."""
        self._wire_session(agent, agent_id=agent_id, agent_type=agent_type)

    async def _finalize_long_memory(self, agent_id: str) -> Optional[dict]:
        """Extract and persist long-term memory from session transcript on loop exit."""
        if agent_id not in self._agents:
            return None

        agent = self._agents[agent_id].agent
        if not getattr(agent, "use_long_memory", False):
            return None
        sess = agent.session
        if sess is None:
            return None

        sess.reload_messages()
        if not sess.messages:
            logger.info(
                "Skip long memory finalize: no session messages agent_id=%s",
                agent_id,
            )
            return None

        uid = agent.user_id or sess.user_id
        long_mem = self._agent_long_memory(agent)
        if uid:
            agent.user_id = uid
            long_mem.user_id = uid

        try:
            result = await long_mem.extract_and_save_from_session(
                sess,
                llm=LLM(config_name="fast"),
            )
            from weakagent.tools.memory.long import _refresh_agent_long_memory

            _refresh_agent_long_memory(agent, long_mem)
            if result.get("should_save"):
                logger.info(
                    "Long memory saved for agent_id=%s type=%s",
                    agent_id,
                    result.get("memory_type"),
                )
            else:
                logger.info("Long memory extraction skipped for agent_id=%s", agent_id)
            return result
        except Exception:
            logger.exception(
                "Failed to finalize long memory for agent_id=%s", agent_id
            )
            return None

    async def _finalize_runtime_session(
        self, agent_id: str, *, status: str = "closed"
    ) -> Optional[str]:
        """Summarize and persist session when an interactive/queue loop ends."""
        if agent_id not in self._agents:
            logger.warning(
                "Skip session finalize: agent_id=%s not in registry", agent_id
            )
            return None

        meta = self._agents[agent_id]
        session = meta.agent.session
        if session is None:
            return None

        session.agent_id = agent_id
        session.agent_type = session.agent_type or meta.agent_type
        try:
            summary = await session.finalize_runtime_summary(
                status=status,
                llm=LLM(config_name="fast"),
                extra={
                    "agent_id": agent_id,
                    "agent_name": getattr(meta.agent, "name", ""),
                    "agent_type": meta.agent_type,
                    "source": "runtime_loop_finalize",
                },
            )
            logger.info(
                "Session finalized session_id=%s agent_id=%s",
                session.session_id,
                agent_id,
            )
            return summary
        except Exception:
            logger.exception(
                "Failed to finalize session agent_id=%s session_id=%s",
                agent_id,
                getattr(session, "session_id", None),
            )
            return None

    def create_agent(
        self,
        agent_type: str,
        *,
        agent_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Get an registered agent from factory and register it in runtime."""
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

    async def cleanup(self, agent_id: str, *, recursive: bool = True) -> None:
        """Cancel tasks, cleanup resources, and remove agent from registry."""
        meta = self.get_meta(agent_id)

        child_ids = list(meta.children) if recursive else []
        for child_id in child_ids:
            if child_id in self._agents:
                await self.cleanup(child_id, recursive=True)

        await self.cancel(agent_id)
        if meta.queue_task and not meta.queue_task.done():
            meta.queue_task.cancel()
            try:
                await meta.queue_task
            except asyncio.CancelledError:
                pass
            meta.queue_task = None

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
