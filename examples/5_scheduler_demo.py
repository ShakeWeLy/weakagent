"""
Scheduler demo covering three flows:

1) Background thread runs SchedulerRunner (periodic scan + dispatch).
2) Task CRUD tools operate on the shared TaskStore.
3) TaskCrudAgent serves CRUD commands through AgentRuntime request/result queues.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from weakagent.agent import AgentFactory, AgentRuntime, AgentSpec, TaskCrudAgent
from weakagent.scheduler import Dispatcher, Scheduler, SchedulerRunner, TaskRegistry, TaskStore
from weakagent.scheduler.executors import DailySummaryExecutor, WeeklyReportExecutor
from weakagent.tools.scheduler import (
    CreateTaskTool,
    DeleteTaskTool,
    GetTaskTool,
    ListTasksTool,
    UpdateTaskTool,
)
from weakagent.tools.scheduler._common import set_task_store
from weakagent.utils.logger import get_logger

logger = get_logger(__name__)


def build_scheduler_stack(store: TaskStore) -> tuple[TaskRegistry, Dispatcher, Scheduler]:
    registry = TaskRegistry()
    registry.register("daily_summary", DailySummaryExecutor)
    registry.register("weekly_report", WeeklyReportExecutor)
    dispatcher = Dispatcher(registry=registry, store=store)
    scheduler = Scheduler(store=store, dispatcher=dispatcher)
    return registry, dispatcher, scheduler


async def demo_thread_scheduler(store: TaskStore, scheduler: Scheduler) -> None:
    """Part 1: run scheduler scan loop in a background thread."""
    store.create_task("daily_summary", payload={"user_id": 1, "topic": "thread-demo"})
    store.create_task("weekly_report", payload={"user_id": 1, "week": "this_week"})

    runner = SchedulerRunner(scheduler, interval=2.0)
    runner.start()
    logger.info("SchedulerRunner started in background thread")

    await asyncio.sleep(6)
    runner.stop()
    logger.info("SchedulerRunner stopped. tasks=%s", [t.id for t in store.list_tasks()])


async def demo_task_tools(store: TaskStore) -> int:
    """Part 2: direct CRUD via scheduler tools."""
    create_tool = CreateTaskTool()
    list_tool = ListTasksTool()
    update_tool = UpdateTaskTool()
    get_tool = GetTaskTool()
    delete_tool = DeleteTaskTool()

    created = await create_tool.execute(
        task_type="daily_summary",
        payload={"source": "tool-demo"},
        priority="high",
    )
    if not created.success or not created.data:
        raise RuntimeError(f"create_task failed: {created.error}")
    task_id = int(created.data["task"]["id"])

    await update_tool.execute(task_id=task_id, payload={"source": "tool-demo-updated"})
    await get_tool.execute(task_id=task_id)
    listed = await list_tool.execute(status="pending")
    logger.info("Tool list_tasks count=%s", listed.data.get("count"))

    await delete_tool.execute(task_id=task_id)
    return int(task_id)


async def demo_agent_queue_crud() -> None:
    """Part 3: TaskCrudAgent handles requests via request queue (no scheduler bridge)."""
    factory = AgentFactory()
    factory.register_spec(
        "task_crud",
        AgentSpec(
            agent_cls=TaskCrudAgent,
            default_kwargs={"name": "task_crud_agent", "verbose": True},
            description="Task CRUD agent for scheduler queue demo",
        ),
    )
    runtime = await AgentRuntime.instance(factory=factory)
    agent_id = runtime.create_agent("task_crud")
    runtime.start_queue_loop(agent_id)

    try:
        runtime.put_request(
            json.dumps(
                {
                    "action": "create",
                    "task_type": "daily_summary",
                    "payload": {"user_id": 99, "via": "request_queue"},
                },
                ensure_ascii=False,
            )
        )
        create_result = await runtime.get_result()
        logger.info("Queue create result: %s", create_result[:200])

        runtime.put_request(json.dumps({"action": "list", "status": "pending"}))
        list_result = await runtime.get_result()
        logger.info("Queue list result: %s", list_result[:300])
    finally:
        await runtime.stop_queue_loop(agent_id)


async def main() -> None:
    demo_db = Path(__file__).with_name("scheduler_demo.sqlite3")
    store = TaskStore(db_path=demo_db)
    set_task_store(store)
    _, _, scheduler = build_scheduler_stack(store)

    logger.info("=== Part 1: thread scheduler ===")
    await demo_thread_scheduler(store, scheduler)

    logger.info("=== Part 2: CRUD tools ===")
    await demo_task_tools(store)

    logger.info("=== Part 3: agent CRUD via request queue ===")
    await demo_agent_queue_crud()

    for task in store.list_tasks():
        logger.info(
            "task id=%s type=%s status=%s next_run_at=%s",
            task.id,
            task.task_type,
            task.status,
            task.next_run_at,
        )


if __name__ == "__main__":
    asyncio.run(main())
