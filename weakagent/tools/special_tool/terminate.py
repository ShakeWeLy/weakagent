from typing import Literal

from weakagent.tools.base import BaseTool, ToolExecutionResult


_TERMINATE_DESCRIPTION = """
Terminate the current interaction.

Use this tool when:
- the user request has been completed
- the assistant cannot continue
- a handoff or final response is needed

Optionally provide a final summary for the user.
"""


class TerminateTool(BaseTool):
    name: str = "terminate"
    description: str = _TERMINATE_DESCRIPTION

    parameters: dict = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["success", "failure"],
                "description": "Final execution status.",
            },
            "message": {
                "type": "string",
                "description": "Final user-facing response or summary.",
            },
            "reason": {
                "type": "string",
                "description": "Internal reason for termination.",
            },
        },
        "required": ["status"],
    }

    async def execute(
        self,
        status: Literal["success", "failure"],
        message: str = "",
        reason: str = "",
    ) -> ToolExecutionResult:
        """
        Terminate the current agent execution.

        Args:
            status: Final execution status.
            message: Final response shown to the user.
            reason: Internal termination reason.

        Returns:
            ToolExecutionResult
        """

        return ToolExecutionResult.ok(
            output=message,
            data={
                "terminate": True,
                "status": status,
                "reason": reason,
            },
        )