"""Apply a text replacement to an existing UTF-8 file (surgical update)."""

from pathlib import Path

from weakagent.tools.base import BaseTool, ToolExecutionResult


class PatchFileTool(BaseTool):
    """Replace exact text in an existing file; use write_file for full rewrites."""

    name: str = "patch_file"
    description: str = (
        "Update an existing UTF-8 file by replacing old_text with new_text. "
        "old_text must match exactly (including whitespace and newlines). "
        "Fails if old_text is missing or appears multiple times unless replace_all is true."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute or relative path to an existing file.",
            },
            "old_text": {
                "type": "string",
                "description": "Exact substring to find in the file.",
            },
            "new_text": {
                "type": "string",
                "description": (
                    "Replacement text (use empty string to remove old_text)."
                ),
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence; if false, exactly one match is required.",
                "default": False,
            },
        },
        "required": ["file_path", "old_text", "new_text"],
    }

    async def execute(
        self,
        file_path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
    ) -> ToolExecutionResult:
        path = Path(file_path)
        if not path.exists():
            return self.fail_response(f"File not found: {path}")
        if not path.is_file():
            return self.fail_response(f"Path is not a file: {path}")

        try:
            raw = path.read_bytes()
        except OSError as e:
            return self.fail_response(f"Could not read file: {e}")

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return self.fail_response(
                "File is not valid UTF-8; patch_file only supports UTF-8 text files."
            )

        occurrences = text.count(old_text)
        if occurrences == 0:
            return self.fail_response(
                "`old_text` not found in file (ensure exact match including newline style)."
            )
        if occurrences > 1 and not replace_all:
            return self.fail_response(
                f"`old_text` matches {occurrences} times; "
                "set replace_all=true to replace all, or narrow old_text."
            )

        try:
            updated = (
                text.replace(old_text, new_text)
                if replace_all
                else text.replace(old_text, new_text, 1)
            )
            path.write_text(updated, encoding="utf-8", newline="\n")
        except OSError as e:
            return self.fail_response(f"Filesystem error writing file: {e}")

        nrep = occurrences if replace_all else 1
        return self.success_response(
            f"Patched file: {path.resolve()} ({nrep} replacement(s))."
        )
