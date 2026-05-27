"""Tenacity retry policy for LLM API calls."""

from __future__ import annotations

from openai import APIError, RateLimitError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from weakagent.utils.exceptions import TokenLimitExceeded

try:
    from openai import APIConnectionError, APITimeoutError
except ImportError:  # pragma: no cover - older SDK stubs
    APIConnectionError = APITimeoutError = ()  # type: ignore[misc, assignment]


def should_retry_llm_call(exc: BaseException) -> bool:
    """Return True only for transient failures (429, 5xx, connection/timeout).

    Client errors such as 400 Bad Request are not retried — the request will not
    succeed on repeat without changing inputs.
    """
    if isinstance(exc, TokenLimitExceeded):
        return False
    if isinstance(exc, ValueError):
        return False
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIError):
        code = getattr(exc, "status_code", None)
        if code == 429:
            return True
        if code is not None and code >= 500:
            return True
        return False
    if APIConnectionError and isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    return False


LLM_API_RETRY = retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception(should_retry_llm_call),
    reraise=True,
)
