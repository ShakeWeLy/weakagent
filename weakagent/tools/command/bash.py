from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, Optional

from weakagent.tools.base import BaseTool, ToolExecutionResult


class BashTool(BaseTool):
    name: str = "bash"
    description: str = (
        "Execute a bash command on the host system. "
        "Returns stdout, stderr, and exit code. "
        "Use for file ops, git, python scripts, package management, etc."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute (e.g. 'ls -la')",
            },
            "cwd": {
                "type": "string",
                "description": (
                    "Working directory for execution (default: project root or current dir)"
                ),
            },
            "timeout": {
                "type": "number",
                "description": "Max execution time in seconds (default: 60, max: 600)",
            },
            "env": {
                "type": "object",
                "description": "Optional extra environment variables (dict of str→str)",
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["command"],
    }

    # Default timeout (overridable per call)
    timeout: int = 60

    # Maximum characters in combined output before truncation
    max_output_length: int = 100_000

    # Blocked patterns to prevent dangerous operations
    blocked_patterns: list[str] = [
        "rm -rf /",
        "rm -rf /*",
        "mkfs.",
        "dd if=",
        "> /dev/",
        "fork()",
        ":(){ :|:& };:",
    ]

    async def execute(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ToolExecutionResult:
        """Execute a real bash command via async subprocess."""
        command = (command or "").strip()
        if not command:
            return ToolExecutionResult.fail("Command is empty")

        # --- Security check ---
        for pattern in self.blocked_patterns:
            if pattern in command:
                return ToolExecutionResult.fail(
                    f"Command blocked: contains disallowed pattern '{pattern}'"
                )

        # --- Resolve working directory ---
        resolved_cwd: Optional[str] = None
        if cwd:
            p = Path(cwd).expanduser().resolve()
            if not p.is_dir():
                return ToolExecutionResult.fail(f"Working directory does not exist: {cwd}")
            resolved_cwd = str(p)

        # --- Merge environment ---
        shell_env = os.environ.copy()
        if env:
            shell_env.update(env)

        # --- Timeout ---
        effective_timeout = min(
            timeout if timeout is not None else self.timeout,
            600,  # hard cap
        )

        # --- Execute ---
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=resolved_cwd,
                env=shell_env,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=effective_timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ToolExecutionResult.fail(
                    f"Command timed out after {effective_timeout}s: {command[:200]}"
                )

        except FileNotFoundError:
            return ToolExecutionResult.fail(
                f"Shell not found — /bin/sh or cmd.exe may not be available"
            )
        except OSError as e:
            return ToolExecutionResult.fail(f"OS error executing command: {e}")

        # --- Decode output ---
        stdout = self._decode(stdout_bytes)
        stderr = self._decode(stderr_bytes)
        exit_code = proc.returncode or 0

        # --- Truncate ---
        combined = stdout
        if stderr:
            combined += "\n--- stderr ---\n" + stderr

        if len(combined) > self.max_output_length:
            combined = combined[: self.max_output_length] + "\n... (truncated)"

        # --- Build result ---
        data: Dict[str, Any] = {
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
        }

        if exit_code == 0:
            return ToolExecutionResult.ok(output=combined, data=data)
        else:
            header = f"Command exited with code {exit_code}"
            return ToolExecutionResult(
                success=False,
                output=f"{header}\n{combined}",
                error=f"{header}: {stderr[:2000] if stderr else combined[:2000]}",
                data=data,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode(data: bytes | None) -> str:
        if data is None:
            return ""
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return data.decode("utf-8", errors="replace")
            except Exception:
                return f"[binary output: {len(data)} bytes]"
