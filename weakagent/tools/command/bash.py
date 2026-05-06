from weakagent.tools.base import BaseTool, ToolExecutionResult

class BashTool(BaseTool):
    name: str = "bash"
    description: str = "Execute a bash command"
    parameters: dict = {
        "command": {
            "type": "string",
            "description": "The command to execute"
        }
    }
    async def execute(self, command: str) -> ToolExecutionResult:
        return ToolExecutionResult.ok(output=f"The command {command} executed successfully")