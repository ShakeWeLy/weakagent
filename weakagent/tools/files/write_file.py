"""Create a new file or overwrite an existing one with full UTF-8 contents."""

from pathlib import Path

from weakagent.tools.base import BaseTool, ToolExecutionResult


class WriteFileTool(BaseTool):
    """Write full file body; creates parent directories when missing."""

    name: str = "write_file"
    description: str = (
        "Create a new file or completely replace an existing file with the given "
        "UTF-8 content. For small localized edits to an existing file, use patch_file instead."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute or relative path to the file.",
            },
            "content": {
                "type": "string",
                "description": "Full file body as UTF-8 (may be empty for an empty file).",
            },
            "create_parents": {
                "type": "boolean",
                "description": "Create parent directories if missing (default true).",
                "default": True,
            },
        },
        "required": ["file_path", "content"],
    }

    async def execute(
        self,
        file_path: str,
        content: str,
        create_parents: bool = True,
    ) -> ToolExecutionResult:
        path = Path(file_path)
        try:
            if create_parents:
                path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
            nbytes = len(content.encode("utf-8"))
            return self.success_response(
                f"Wrote file: {path.resolve()} ({nbytes} bytes UTF-8)."
            )
        except OSError as e:
            return self.fail_response(f"Filesystem error for {path}: {e}")
