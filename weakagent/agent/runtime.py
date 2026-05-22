import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from weakagent.agent.base import BaseAgent
from weakagent.agent.factory import AgentFactory
from weakagent.llm.llm import LLM
from weakagent.schemas.agent import AgentState
from weakagent.utils.logger import logger
from weakagent.adapters.input import BaseInputSource, CLIInput
from weakagent.adapters.output import BaseOutputSource, CLIOutput
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

        self._wire_runtime_session(agent, agent_id=resolved_id, agent_type=resolved_type)
        return resolved_id

    def _wire_runtime_session(
        self, agent: BaseAgent, *, agent_id: str, agent_type: str
    ) -> None:
        """Bind managed agent metadata to runtime_memory session rows."""
        rm = getattr(agent, "runtime_memory", None)
        if rm is None:
            return
        rm.agent_id = agent_id
        rm.agent_type = rm.agent_type or agent_type
        try:
            rm.ensure_session()
        except Exception:
            logger.exception("Failed to ensure runtime session for agent_id=%s", agent_id)

    async def _finalize_runtime_session(
        self, agent_id: str, *, status: str = "closed"
    ) -> Optional[str]:
        """Summarize and persist runtime session when an interactive/queue loop ends."""
        if agent_id not in self._agents:
            logger.warning(
                "Skip runtime session finalize: agent_id=%s not in registry", agent_id
            )
            return None
        meta = self._agents[agent_id]
        agent = meta.agent
        rm = getattr(agent, "runtime_memory", None)
        if rm is None:
            return None
        if not rm.messages:
            logger.info("Skip runtime session finalize: no messages for agent_id=%s", agent_id)
            return None

        rm.agent_id = agent_id
        rm.agent_type = rm.agent_type or meta.agent_type
        try:
            summary = await rm.finalize_session(
                status=status,
                run_id=f"loop_{agent_id}",
                llm=LLM(config_name="fast"),
                extra={
                    "agent_id": agent_id,
                    "agent_name": getattr(agent, "name", ""),
                    "agent_type": meta.agent_type,
                },
            )
            logger.info(
                "Runtime session finalized session_id=%s agent_id=%s",
                rm.session_id,
                agent_id,
            )
            return summary
        except Exception:
            logger.exception(
                "Failed to finalize runtime session agent_id=%s session_id=%s",
                agent_id,
                getattr(rm, "session_id", None),
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

    # ===============synchronous mode=================
    async def run(
        self,
        agent_id: str,
        request: Optional[str] = None,
        *,
        input_source: Optional[BaseInputSource] = None,
        output_source: Optional[BaseOutputSource] = None,
        emit_output: bool = True,
    ) -> str:
        """Run an agent synchronously (await until completion)."""
        inp = input_source or CLIInput()
        out = output_source or CLIOutput()
        agent = self.get(agent_id)
        if request:
            logger.debug(f"User request: {request}")
        else:
            request = await inp.read()
            logger.debug(f"User request: {request}")
        if not (request or "").strip():
            return ""
        result = await agent.run(request=request)
        if emit_output and result:
            out.dispatch(result)
        return result

    # ===============background mode=================
    def run_in_background(
        self,
        agent_id: str,
        request: Optional[str] = None,
        *,
        input_source: Optional[BaseInputSource] = None,
        output_source: Optional[BaseOutputSource] = None,
        emit_output: bool = False,
    ) -> asyncio.Task[str]:
        """Schedule ``run()`` on the event loop and return its Task.

        Prefer passing ``request`` explicitly. If omitted, ``run()`` falls back to
        ``input_source`` (default CLI) inside the background task — rarely desirable
        for CLI; use ``APIInput`` or a pre-filled request instead.
        """
        meta = self.get_meta(agent_id)
        if meta.task and not meta.task.done():
            raise RuntimeError(f"agent already running: {agent_id}")

        async def _runner() -> str:
            return await self.run(
                agent_id,
                request=request,
                input_source=input_source,
                output_source=output_source,
                emit_output=emit_output,
            )

        task: asyncio.Task[str] = asyncio.create_task(
            _runner(), name=f"bg-run-{agent_id}"
        )
        meta.task = task

        def _clear_task_ref(t: asyncio.Task[str]) -> None:
            if meta.task is t:
                meta.task = None

        task.add_done_callback(_clear_task_ref)
        return task
    
    # ===============interactive mode=================
    async def run_loop(
        self,
        agent_id: str,
        request: Optional[str] = None,
        *,
        input_source: Optional[BaseInputSource] = None,
        output_source: Optional[BaseOutputSource] = None,
    ):
        """Interactive loop: read via input_source until exit/quit/q."""
        inp = input_source or CLIInput()
        out = output_source or CLIOutput()
        pending: Optional[str] = request
        try:
            while True:
                if pending is not None:
                    current = pending
                    pending = None
                else:
                    current = await inp.read()
                logger.debug(f"User request: {current}")

                if not (current or "").strip():
                    continue
                if current.lower() in {"exit", "quit", "q"}:
                    break
                await self.run(
                    agent_id,
                    request=current,
                    input_source=inp,
                    output_source=out,
                )
        finally:
            await self._finalize_runtime_session(agent_id)
            await self.cleanup(agent_id)
            logger.info("Cleanup complete.")
    
    # ===============queue mode=================
    # Producer: put_request() at any time (main thread, scheduler thread, stdin, HTTP, ...).
    # Consumer: start_queue_loop() runs a long-lived task that takes one request at a time,
    # runs agent.run() (LLM / tools), pushes the result, then immediately takes the next
    # queued request if the queue is not empty.
    def run_queue_loop(self, agent_id: str) -> asyncio.Task:
        """Start the queue consumer in the background if not already running."""
        meta = self.get_meta(agent_id)
        if meta.queue_task and not meta.queue_task.done():
            return meta.queue_task
        meta.queue_task = asyncio.create_task(
            self.run_loop_async(agent_id),
            name=f"queue-loop-{agent_id}",
        )
        logger.info("Queue loop started for agent_id=%s", agent_id)
        return meta.queue_task

    def put_request(self, request: str) -> None:
        """Enqueue one request (thread-safe). Ignores blank strings."""
        text = (request or "").strip()
        if not text:
            return
        loop = self._loop
        if loop is None:
            self.request_queue.put_nowait(text)
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self.request_queue.put_nowait(text)
        else:
            asyncio.run_coroutine_threadsafe(self.request_queue.put(text), loop)

    async def get_result(self) -> str:
        """Await the next result (FIFO, one per processed request)."""
        return await self.result_queue.get()

    def start_queue_loop(self, agent_id: str) -> asyncio.Task:
        """Start the queue consumer in the background if not already running."""
        meta = self.get_meta(agent_id)
        if meta.queue_task and not meta.queue_task.done():
            return meta.queue_task
        meta.queue_task = asyncio.create_task(
            self.run_loop_async(agent_id),
            name=f"queue-loop-{agent_id}",
        )
        logger.info("Queue loop started for agent_id=%s", agent_id)
        return meta.queue_task
    
    async def stop_queue_loop(self, agent_id: str) -> None:
        """Stop the queue consumer by enqueueing exit and awaiting its task."""
        meta = self.get_meta(agent_id)
        if not meta.queue_task or meta.queue_task.done():
            return
        self.put_request("exit")
        try:
            await meta.queue_task
        except asyncio.CancelledError:
            pass
        meta.queue_task = None

    def is_queue_loop_running(self, agent_id: str) -> bool:
        meta = self.get_meta(agent_id)
        return meta.queue_task is not None and not meta.queue_task.done()

    async def _process_one_request(self, agent_id: str, request: str) -> str:
        """Run a single queued request and return the agent output."""
        meta = self.get_meta(agent_id)
        meta.agent.current_step = 0
        meta.agent.state = AgentState.IDLE
        return await self.run(agent_id, request=request)

    async def run_loop_async(self, agent_id: str) -> None:
        """Consume request_queue: one request -> one run -> one result; drain backlog."""
        try:
            while True:
                request = await self.request_queue.get()
                while request is not None:
                    if request.lower() in {"exit", "quit", "q"}:
                        return
                    logger.info("Queue processing request (len=%s)", len(request))
                    result = await self._process_one_request(agent_id, request)
                    await self.result_queue.put(result)
                    try:
                        request = self.request_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        request = None
        finally:
            await self._finalize_runtime_session(agent_id)
            try:
                meta = self.get_meta(agent_id)
                meta.queue_task = None
            except KeyError:
                pass
            logger.info("run_loop_async finished for agent_id=%s", agent_id)


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
        finally:
            meta.task = None
        return True


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
