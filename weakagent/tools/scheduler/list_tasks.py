from __future__ import annotations

from typing import Optional

from weakagent.scheduler import TaskStatus
from weakagent.tools.base import BaseTool, ToolExecutionResult
from weakagent.tools.scheduler._common import get_task_store, task_to_dict


class ListTasksTool(BaseTool):
    name: str = "list_tasks"
    description: str = "List scheduled tasks, optionally filtered by status."
    parameters: dict = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["pending", "running", "completed", "failed", "cancelled"],
                "description": "Optional status filter.",
            },
        },
    }

    async def execute(self, status: Optional[str] = None) -> ToolExecutionResult:
        store = get_task_store()
        status_filter = None
        if status:
            try:
                status_filter = TaskStatus(status.lower())
            except ValueError:
                return self.fail_response(f"Unknown status: {status}")

        try:
            tasks = store.list_tasks(status=status_filter)
        except Exception as exc:
            return self.fail_response(f"Failed to list tasks: {exc}")

        return self.success_response(
            {
                "count": len(tasks),
                "tasks": [task_to_dict(t) for t in tasks],
            }
        )
