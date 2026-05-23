import json
import re
from typing import Any, Dict, List, Optional, Union


def extract_json_substring(text: str) -> str:
    """Extract a JSON object/array substring from LLM text.

    Supports markdown ```json fences, raw JSON, or the first embedded `{...}` / `[...]`.
    """
    text = (text or "").strip()
    if not text:
        return ""
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        return fence.group(1).strip()
    if text.startswith("{") or text.startswith("["):
        return text
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    return m.group(1) if m else text


def json_to_dict(json_str: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON string into a dict; returns None on failure or non-object JSON."""
    sub = extract_json_substring(json_str)
    if not sub:
        return None
    try:
        data = json.loads(sub)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_json_from_text(text: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """Parse JSON object or array from LLM output text."""
    sub = extract_json_substring(text)
    if not sub:
        return None
    try:
        return json.loads(sub)
    except json.JSONDecodeError:
        return None


def parse_llm_json_dict(text: str) -> Optional[Dict[str, Any]]:
    """Parse LLM output and return a dict, or None if not a JSON object."""
    data = parse_json_from_text(text)
    return data if isinstance(data, dict) else None
