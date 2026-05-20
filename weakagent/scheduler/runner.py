from __future__ import annotations

import asyncio
import threading
from typing import Optional

from weakagent.scheduler.task_manager import Scheduler
from weakagent.utils.logger import get_logger

logger = get_logger(__name__)


def run_scheduler_once(scheduler: Scheduler, *, limit: int = 50) -> int:
    """Run one scheduler scan round from a worker thread."""

    async def _once() -> int:
        return await scheduler.run_once(limit=limit)

    return asyncio.run(_once())


class SchedulerRunner:
    """Background thread that periodically scans and dispatches due tasks."""

    def __init__(self, scheduler: Scheduler, *, interval: float = 5.0) -> None:
        self.scheduler = scheduler
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, *, daemon: bool = True) -> None:
        """Start the scheduler loop in a background thread."""
        if self.is_running:
            raise RuntimeError("SchedulerRunner is already running")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="scheduler-runner", daemon=daemon)
        self._thread.start()
        logger.info("SchedulerRunner started. interval=%ss", self.interval)

    def stop(self, *, timeout: Optional[float] = None) -> None:
        """Signal the loop to stop and wait for the worker thread."""
        if not self._thread:
            return
        self._stop.set()
        # Allow in-flight dispatches to finish before giving up on join.
        wait_for = timeout if timeout is not None else max(self.interval + 5.0, 30.0)
        self._thread.join(timeout=wait_for)
        if self._thread.is_alive():
            logger.warning("SchedulerRunner thread did not stop within timeout")
        else:
            logger.info("SchedulerRunner stopped")
        self._thread = None

    def _run(self) -> None:
        async def _loop() -> None:
            while not self._stop.is_set():
                try:
                    count = await self.scheduler.run_once(limit=50)
                    if count:
                        logger.info("SchedulerRunner dispatched %d task(s)", count)
                except Exception:
                    logger.exception("SchedulerRunner loop failed")
                # Sleep in small slices so stop() is responsive.
                elapsed = 0.0
                while elapsed < self.interval and not self._stop.is_set():
                    await asyncio.sleep(min(0.5, self.interval - elapsed))
                    elapsed += 0.5

        asyncio.run(_loop())
