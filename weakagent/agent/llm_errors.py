"""Shared LLM failure handling for agent think/act loops."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from openai import APIError

from weakagent.schemas.agent import AgentState
from weakagent.utils.exceptions import TokenLimitExceeded
from weakagent.utils.logger import get_logger
from weakagent.utils.run_errors import format_run_error, unwrap_run_exception

if TYPE_CHECKING:
    from weakagent.agent.base import BaseAgent

logger = get_logger(__name__)


def handle_llm_think_error(agent: BaseAgent, exc: Exception) -> Optional[bool]:
    """Handle LLM errors during think().

    Returns:
        False if the error was handled and the step should stop cleanly.
        None if the caller should re-raise (validation or unknown errors).
    """
    root = unwrap_run_exception(exc)

    if isinstance(root, TokenLimitExceeded):
        logger.error("Token limit error during think: %s", root)
        agent.update_memory(
            "assistant",
            f"Maximum token limit reached, cannot continue execution: {root}",
        )
        agent.last_result = str(root)
        agent.state = AgentState.FINISHED
        return False

    if isinstance(root, ValueError):
        return None

    if isinstance(root, APIError):
        code = getattr(root, "status_code", None)
        if code is not None and 400 <= code < 500 and code != 429:
            msg = format_run_error(root)
            logger.error("LLM client error during think (HTTP %s): %s", code, msg)
            agent.update_memory("assistant", f"LLM API client error: {msg}")
            agent.last_result = msg
            if not agent.schedule_recovery_request(msg):
                agent.last_result = f"{msg} (auto-recovery budget exhausted)"
            agent.state = AgentState.FINISHED
            return False

    return None
