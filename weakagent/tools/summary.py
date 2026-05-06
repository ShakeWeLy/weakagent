from weakagent.tools.base import BaseTool, ToolExecutionResult

_SUMMARY_DESCRIPTION = """Summarize the current interaction, progress, findings, or collected information.
Use this tool when the user explicitly asks for a summary, or when a concise recap is needed before termination or handoff."""


class Summary(BaseTool):
    name: str = "summary"
    description: str = _SUMMARY_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The main content to summarize."
            },
            "style": {
                "type": "string",
                "description": "Summary style.",
                "enum": ["brief", "detailed", "bullet"],
                "default": "brief"
            }
        },
        "required": ["content"],
    }

    async def execute(self, content: str, style: str = "brief") -> ToolExecutionResult:
        """Return a formatted summary request result"""

        if style == "bullet":
            lines = [line.strip() for line in content.split(".") if line.strip()]
            return self.success_response("\n".join(f"- {line}" for line in lines))

        if style == "detailed":
            return self.success_response(f"Detailed Summary:\n{content}")

        return self.success_response(f"Summary: {content}")