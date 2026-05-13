from weakagent.tools.base import BaseTool, ToolExecutionResult


_ASK_HUMAN_DESCRIPTION = """
Request additional input or clarification from the user.

Use this tool when:
- required information is missing
- the task is ambiguous
- user confirmation is needed
- human intervention is required to continue
"""


class AskHumanTool(BaseTool):
    name: str = "ask_human"
    description: str = _ASK_HUMAN_DESCRIPTION

    parameters: dict = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "Question or clarification request for the user."
            }
        },
        "required": ["question"],
    }

    async def execute(self, question: str) -> ToolExecutionResult:
        return ToolExecutionResult.ok(
            output=question,
            data={"await_human": True},
        )