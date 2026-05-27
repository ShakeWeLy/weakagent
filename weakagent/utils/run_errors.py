"""Format agent/LLM run failures and build auto-recovery user prompts."""

from __future__ import annotations

import hashlib

# Max auto-recovery turns per identical error fingerprint within one user turn chain.
MAX_RECOVERY_PER_FINGERPRINT = 2

RECOVERY_REQUEST_PREFIX = "[Runtime error from previous turn — auto-injected]"


def unwrap_run_exception(exc: BaseException) -> BaseException:
    """Unwrap tenacity RetryError to the underlying API exception."""
    if exc.__class__.__name__ == "RetryError" and exc.__cause__ is not None:
        return exc.__cause__
    return exc


def format_run_error(exc: BaseException) -> str:
    """Single-line error summary for logs and recovery prompts."""
    root = unwrap_run_exception(exc)
    return f"{type(root).__name__}: {root}"


def error_fingerprint(error_text: str) -> str:
    """Stable short hash for deduplicating recovery attempts."""
    text = (error_text or "").strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def is_recovery_request(text: str) -> bool:
    return (text or "").strip().startswith(RECOVERY_REQUEST_PREFIX)


def build_recovery_request(error_text: str) -> str:
    """Build the next-turn user message injected after a failed run."""
    return (
        f"{RECOVERY_REQUEST_PREFIX}\n"
        f"{error_text}\n\n"
        "Diagnose the cause, fix tool or config issues if needed, then continue "
        "the original task."
    )
