from weakagent.tools.base import BaseTool, ToolExecutionResult

_TERMINATE_DESCRIPTION = """Terminate the interaction when the request is met OR if the assistant cannot proceed further with the task.
When you have finished all the tasks, call this tool to end the work."""


class Terminate(BaseTool):
    name: str = "terminate"
    description: str = _TERMINATE_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "The finish status of the interaction.",
                "enum": ["success", "failure"],
            },
            "reason": {
                "type": "string",
                "description": "The reason for the termination.",
            }
        },
        "required": ["status"],
    }

    async def execute(self, status: str, reason: str = "") -> ToolExecutionResult:
        """Finish the current execution"""
        return self.success_response(f"The interaction has been completed with status: {status}, reason: {reason}")
