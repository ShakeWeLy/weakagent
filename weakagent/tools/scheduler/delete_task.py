from __future__ import annotations

from weakagent.tools.base import BaseTool, ToolExecutionResult
from weakagent.tools.scheduler._common import get_task_store, task_to_dict


class DeleteTaskTool(BaseTool):
    name: str = "delete_task"
    description: str = (
        "Delete a task permanently, or cancel it (set status=cancelled) when cancel_only=true."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "task_id": {"type": "integer", "description": "Task id to delete or cancel."},
            "cancel_only": {
                "type": "boolean",
                "description": "If true, mark cancelled instead of deleting the row.",
                "default": False,
            },
        },
        "required": ["task_id"],
    }

    async def execute(
        self,
        task_id: int,
        cancel_only: bool = False,
    ) -> ToolExecutionResult:
        store = get_task_store()
        try:
            if cancel_only:
                task = store.cancel_task(int(task_id))
                return self.success_response(
                    {"message": "Task cancelled", "task": task_to_dict(task)}
                )
            deleted = store.delete_task(int(task_id))
        except KeyError:
            return self.fail_response(f"task_id not found: {task_id}")
        except Exception as exc:
            return self.fail_response(f"Failed to delete task: {exc}")

        if not deleted:
            return self.fail_response(f"task_id not found: {task_id}")

        return self.success_response(
            {"message": "Task deleted", "task_id": int(task_id)}
        )
