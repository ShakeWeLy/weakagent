from __future__ import annotations

from typing import Any, Dict, Optional

from weakagent.scheduler import TaskPriority
from weakagent.tools.base import BaseTool, ToolExecutionResult
from weakagent.tools.scheduler._common import get_task_store, parse_next_run_at, task_to_dict


class UpdateTaskTool(BaseTool):
    name: str = "update_task"
    description: str = (
        "Update an existing task's payload, priority, or next_run_at. "
        "Only provided fields are changed."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "task_id": {"type": "integer", "description": "Task id to update."},
            "payload": {"type": "object", "description": "New payload object."},
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high", "critical"],
            },
            "next_run_at": {
                "type": "string",
                "description": "New ISO-8601 UTC due time.",
            },
        },
        "required": ["task_id"],
    }

    _PRIORITY_MAP = {
        "low": TaskPriority.LOW,
        "medium": TaskPriority.MEDIUM,
        "high": TaskPriority.HIGH,
        "critical": TaskPriority.CRITICAL,
    }

    async def execute(
        self,
        task_id: int,
        payload: Optional[Dict[str, Any]] = None,
        priority: Optional[str] = None,
        next_run_at: Optional[str] = None,
    ) -> ToolExecutionResult:
        store = get_task_store()
        prio = None
        if priority is not None:
            prio = self._PRIORITY_MAP.get(priority.lower())
            if prio is None:
                return self.fail_response(f"Unknown priority: {priority}")

        try:
            due_at = parse_next_run_at(next_run_at) if next_run_at is not None else None
            if next_run_at is not None and due_at is None and next_run_at.strip():
                return self.fail_response("next_run_at cannot be empty when provided")
        except ValueError as exc:
            return self.fail_response(f"Invalid next_run_at: {exc}")

        try:
            task = store.update_task(
                int(task_id),
                payload=payload,
                priority=prio,
                next_run_at=due_at if next_run_at is not None else None,
            )
        except KeyError:
            return self.fail_response(f"task_id not found: {task_id}")
        except Exception as exc:
            return self.fail_response(f"Failed to update task: {exc}")

        return self.success_response(
            {"message": "Task updated", "task": task_to_dict(task)}
        )
