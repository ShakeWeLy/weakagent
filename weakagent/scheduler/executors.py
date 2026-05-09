from __future__ import annotations

from datetime import timedelta

from weakagent.scheduler.task_manager import Executor, Task, TaskStore, _utc_now
from weakagent.utils.logger import get_logger

logger = get_logger(__name__)


class DailySummaryExecutor(Executor):
    async def execute(self, task: Task, store: TaskStore) -> None:
        # Do your real work here (agent/tool/memory/etc.)
        logger.info("daily_summary executing. task_id=%s payload=%s", task.id, task.payload)

        # Re-schedule for next day (keeps it periodic).
        store.mark_pending(task.id, next_run_at=_utc_now() + timedelta(days=1))


class WeeklyReportExecutor(Executor):
    async def execute(self, task: Task, store: TaskStore) -> None:
        logger.info("weekly_report executing. task_id=%s payload=%s", task.id, task.payload)

        # Re-schedule for next week (keeps it periodic).
        store.mark_pending(task.id, next_run_at=_utc_now() + timedelta(days=7))

