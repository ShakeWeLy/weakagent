"""Validate and normalize OpenAI function-calling parameter schemas."""

from __future__ import annotations

from typing import Any, Dict

from weakagent.utils.logger import get_logger

logger = get_logger(__name__)


def normalize_tool_parameters(tool_name: str, schema: dict) -> dict:
    """Return a copy of ``schema`` safe to send as function ``parameters``.

    Raises:
        ValueError: When ``properties`` is missing or not a dict (cannot auto-fix).
    """
    if not isinstance(schema, dict):
        raise ValueError(
            f"Tool {tool_name!r}: parameters must be a dict, got {type(schema).__name__}"
        )

    normalized = dict(schema)

    properties = normalized.get("properties")
    if properties is None or not isinstance(properties, dict):
        raise ValueError(
            f"Tool {tool_name!r}: missing valid 'properties' object in parameters schema."
        )

    if normalized.get("type") != "object":
        logger.error(
            "Tool %r parameters.type must be 'object' (got %r); coercing locally.",
            tool_name,
            normalized.get("type"),
        )
        normalized["type"] = "object"

    return normalized
