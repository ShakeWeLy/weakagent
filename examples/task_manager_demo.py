"""
Demo for the new scheduler architecture:

TaskRegistry -> TaskStore(sqlite) -> Scheduler(scan) -> Dispatcher(route) -> Executor(execute)
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from weakagent.scheduler import Dispatcher, Scheduler, TaskRegistry, TaskStore
from weakagent.scheduler.executors import DailySummaryExecutor, WeeklyReportExecutor


async def main() -> None:
    # 1) Registry: runtime task_type -> executor
    registry = TaskRegistry()
    registry.register("daily_summary", DailySummaryExecutor)
    registry.register("weekly_report", WeeklyReportExecutor)

    # 2) Store: sqlite persistence (data only)
    demo_db = Path(__file__).with_name("weakagent.sqlite3")
    store = TaskStore(db_path=demo_db)

    # 3) Dispatcher: route task -> executor -> execute
    dispatcher = Dispatcher(registry=registry, store=store)

    # 4) Scheduler: scan due tasks, then dispatch
    scheduler = Scheduler(store=store, dispatcher=dispatcher)

    # Seed tasks (due immediately)
    t1 = store.create_task("daily_summary", payload={"user_id": 1, "topic": "yesterday"})
    t2 = store.create_task("weekly_report", payload={"user_id": 1, "week": "this_week"})

    print("Seeded tasks:", [t1.id, t2.id])
    print("Supported types:", registry.supported_types)

    # Run one scan round (dispatch due tasks once)
    executed = await scheduler.run_once(limit=50)
    print("Scheduler executed count:", executed)

    # Show resulting rows (executors re-schedule them as PENDING with next_run_at)
    tasks = store.list_tasks()
    for t in tasks:
        print(
            f"task_id={t.id} type={t.task_type} status={t.status} attempts={t.attempts} next_run_at={t.next_run_at}"
        )


if __name__ == "__main__":
    asyncio.run(main())

