"""Agent execution modes: sync, background, interactive, and queue."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Optional

from weakagent.adapters.input import BaseInputSource, CLIInput
from weakagent.adapters.output import BaseOutputSource, CLIOutput
from weakagent.schemas.agent import AgentState
from weakagent.utils.logger import logger

if TYPE_CHECKING:
    from weakagent.agent.runtime import AgentRuntime


class AgentRunMixin:
    """Mixin providing run / loop / queue execution on :class:`AgentRuntime`.

    Expects the host class to expose registry access (``get``, ``get_meta``),
    session helpers (``_load_last_runtime_session``, ``_finalize_*``),
    ``cleanup``, and queue fields (``request_queue``, ``result_queue``, ``_loop``).
    """

    # =============== synchronous mode ===============
    async def run(
        self: AgentRuntime,
        agent_id: str,
        request: Optional[str] = None,
        *,
        input_source: Optional[BaseInputSource] = None,
        output_source: Optional[BaseOutputSource] = None,
        emit_output: bool = True,
        load_last_session: bool = False,
        last_session_messages: int = 10,
        use_long_memory: bool = False,
    ) -> str:
        """Run an agent synchronously (await until completion)."""
        self._load_last_runtime_session(
            agent_id,
            load_last_session=load_last_session,
            last_session_messages=last_session_messages,
        )
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
        result = await agent.run(request=request, use_long_memory=use_long_memory)
        if emit_output and result:
            out.dispatch(result)
        return result

    # =============== background mode ===============
    def run_in_background(
        self: AgentRuntime,
        agent_id: str,
        request: Optional[str] = None,
        *,
        input_source: Optional[BaseInputSource] = None,
        output_source: Optional[BaseOutputSource] = None,
        emit_output: bool = False,
        load_last_session: bool = False,
        last_session_messages: int = 10,
        use_long_memory: bool = False,
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
                load_last_session=load_last_session,
                last_session_messages=last_session_messages,
                use_long_memory=use_long_memory,
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

    # =============== interactive mode ===============
    async def run_loop(
        self: AgentRuntime,
        agent_id: str,
        request: Optional[str] = None,
        *,
        input_source: Optional[BaseInputSource] = None,
        output_source: Optional[BaseOutputSource] = None,
        load_last_session: bool = True,
        last_session_messages: int = 10,
        use_long_memory: bool = False,
    ):
        """Interactive loop: read via input_source until exit/quit/q."""
        self._load_last_runtime_session(
            agent_id,
            load_last_session=load_last_session,
            last_session_messages=last_session_messages,
        )
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
                    use_long_memory=use_long_memory,
                )
        finally:
            await self._finalize_runtime_session(agent_id)
            await self._finalize_long_memory(agent_id, use_long_memory=use_long_memory)
            await self.cleanup(agent_id)
            logger.info("Cleanup complete.")

    # =============== queue mode ===============
    # Producer: put_request() at any time (main thread, scheduler, stdin, HTTP, ...).
    # Consumer: start_queue_loop() runs a long-lived task that processes one request
    # at a time, pushes the result, then drains any backlog before blocking again.
    def run_queue_loop(
        self: AgentRuntime,
        agent_id: str,
        *,
        load_last_session: bool = False,
        last_session_messages: int = 10,
        use_long_memory: bool = False,
    ) -> asyncio.Task:
        """Start the queue consumer in the background if not already running."""
        meta = self.get_meta(agent_id)
        if meta.queue_task and not meta.queue_task.done():
            return meta.queue_task
        meta.queue_task = asyncio.create_task(
            self.run_loop_async(
                agent_id,
                load_last_session=load_last_session,
                last_session_messages=last_session_messages,
                use_long_memory=use_long_memory,
            ),
            name=f"queue-loop-{agent_id}",
        )
        logger.info("Queue loop started for agent_id=%s", agent_id)
        return meta.queue_task

    def put_request(self: AgentRuntime, request: str) -> None:
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

    async def get_result(self: AgentRuntime) -> str:
        """Await the next result (FIFO, one per processed request)."""
        return await self.result_queue.get()

    def start_queue_loop(
        self: AgentRuntime,
        agent_id: str,
        *,
        load_last_session: bool = False,
        last_session_messages: int = 10,
        use_long_memory: bool = False,
    ) -> asyncio.Task:
        """Start the queue consumer in the background if not already running."""
        return self.run_queue_loop(
            agent_id,
            load_last_session=load_last_session,
            last_session_messages=last_session_messages,
            use_long_memory=use_long_memory,
        )

    async def stop_queue_loop(self: AgentRuntime, agent_id: str) -> None:
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

    def is_queue_loop_running(self: AgentRuntime, agent_id: str) -> bool:
        meta = self.get_meta(agent_id)
        return meta.queue_task is not None and not meta.queue_task.done()

    async def _process_one_request(
        self: AgentRuntime,
        agent_id: str,
        request: str,
        *,
        use_long_memory: bool = False,
    ) -> str:
        """Run a single queued request and return the agent output."""
        meta = self.get_meta(agent_id)
        meta.agent.current_step = 0
        meta.agent.state = AgentState.IDLE
        return await self.run(
            agent_id,
            request=request,
            use_long_memory=use_long_memory,
        )

    async def run_loop_async(
        self: AgentRuntime,
        agent_id: str,
        *,
        load_last_session: bool = False,
        last_session_messages: int = 10,
        use_long_memory: bool = False,
    ) -> None:
        """Consume request_queue: one request -> one run -> one result; drain backlog."""
        self._load_last_runtime_session(
            agent_id,
            load_last_session=load_last_session,
            last_session_messages=last_session_messages,
        )
        try:
            while True:
                request = await self.request_queue.get()
                while request is not None:
                    if request.lower() in {"exit", "quit", "q"}:
                        return
                    logger.info("Queue processing request (len=%s)", len(request))
                    result = await self._process_one_request(
                        agent_id,
                        request,
                        use_long_memory=use_long_memory,
                    )
                    await self.result_queue.put(result)
                    try:
                        request = self.request_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        request = None
        finally:
            await self._finalize_runtime_session(agent_id)
            await self._finalize_long_memory(agent_id, use_long_memory=use_long_memory)
            try:
                meta = self.get_meta(agent_id)
                meta.queue_task = None
            except KeyError:
                pass
            logger.info("run_loop_async finished for agent_id=%s", agent_id)

    async def cancel(self: AgentRuntime, agent_id: str) -> bool:
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
