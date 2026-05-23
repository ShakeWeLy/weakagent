from pathlib import Path
from typing import Optional

from weakagent.tools.base import BaseTool, ToolExecutionResult


class ReadFileTool(BaseTool):
    """Read text file contents (e.g. SKILL.md for agent skills)."""

    name: str = "read"
    description: str = (
        "Read a text file from disk. Use to load SKILL.md or other skill resources "
        "listed in <available_skills>."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute or project-relative path to the file",
            },
            "offset": {
                "type": "integer",
                "description": "0-based line offset to start reading (default 0)",
                "default": 0,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read (default 500)",
                "default": 500,
            },
        },
        "required": ["file_path"],
    }

    async def execute(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 500,
    ) -> ToolExecutionResult:
        if not file_path or not str(file_path).strip():
            return self.fail_response("file_path is required")

        path = Path(file_path).expanduser()
        if not path.is_absolute():
            from weakagent.config.settings import PROJECT_ROOT

            path = (PROJECT_ROOT / path).resolve()

        if not path.exists():
            return self.fail_response(f"File not found: {path}")
        if not path.is_file():
            return self.fail_response(f"Not a file: {path}")

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return self.fail_response(f"Could not read file: {e}")

        lines = text.splitlines()
        offset = max(0, int(offset))
        limit = max(1, min(int(limit), 5000))
        slice_lines = lines[offset : offset + limit]

        header = f"File: {path}\nLines {offset + 1}-{offset + len(slice_lines)} of {len(lines)}\n\n"
        body = "\n".join(slice_lines)
        if offset + limit < len(lines):
            body += f"\n\n... ({len(lines) - offset - limit} more lines)"

        return self.success_response(header + body)
