from __future__ import annotations

from weakagent.tools.base import BaseTool, ToolExecutionResult
from weakagent.tools.scheduler._common import get_task_store, task_to_dict


class GetTaskTool(BaseTool):
    name: str = "get_task"
    description: str = "Read a single scheduled task by id."
    parameters: dict = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "integer",
                "description": "Primary key of the task row.",
            },
        },
        "required": ["task_id"],
    }

    async def execute(self, task_id: int) -> ToolExecutionResult:
        store = get_task_store()
        try:
            task = store.get_task(int(task_id))
        except KeyError:
            return self.fail_response(f"task_id not found: {task_id}")
        except Exception as exc:
            return self.fail_response(f"Failed to read task: {exc}")

        return self.success_response({"task": task_to_dict(task)})
