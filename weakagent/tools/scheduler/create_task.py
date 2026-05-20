from __future__ import annotations

from typing import Any, Dict, Optional

from weakagent.scheduler import TaskPriority
from weakagent.tools.base import BaseTool, ToolExecutionResult
from weakagent.tools.scheduler._common import get_task_store, parse_next_run_at, task_to_dict


class CreateTaskTool(BaseTool):
    name: str = "create_task"
    description: str = (
        "Create a new scheduled task in TaskStore. "
        "Use task_type to route execution; payload holds task parameters."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "task_type": {
                "type": "string",
                "description": "Executor routing key, e.g. daily_summary or weekly_report.",
            },
            "payload": {
                "type": "object",
                "description": "JSON-serializable task parameters.",
            },
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high", "critical"],
                "description": "Task priority (default medium).",
            },
            "next_run_at": {
                "type": "string",
                "description": "Optional ISO-8601 UTC time when the task becomes due.",
            },
        },
        "required": ["task_type"],
    }

    _PRIORITY_MAP = {
        "low": TaskPriority.LOW,
        "medium": TaskPriority.MEDIUM,
        "high": TaskPriority.HIGH,
        "critical": TaskPriority.CRITICAL,
    }

    async def execute(
        self,
        task_type: str,
        payload: Optional[Dict[str, Any]] = None,
        priority: str = "medium",
        next_run_at: Optional[str] = None,
    ) -> ToolExecutionResult:
        if not task_type:
            return self.fail_response("task_type is required")

        prio = self._PRIORITY_MAP.get((priority or "medium").lower())
        if prio is None:
            return self.fail_response(f"Unknown priority: {priority}")

        try:
            due_at = parse_next_run_at(next_run_at)
        except ValueError as exc:
            return self.fail_response(f"Invalid next_run_at: {exc}")

        store = get_task_store()
        try:
            task = store.create_task(
                task_type,
                payload=payload or {},
                priority=prio,
                next_run_at=due_at,
            )
        except Exception as exc:
            return self.fail_response(f"Failed to create task: {exc}")

        return self.success_response(
            {"message": "Task created", "task": task_to_dict(task)}
        )
