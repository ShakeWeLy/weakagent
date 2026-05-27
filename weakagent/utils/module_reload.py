"""Safe importlib.reload helpers for development hot-reload."""

from __future__ import annotations

import importlib
import sys
from typing import Any, Dict, Iterable, List, Optional

_ALLOWED_PREFIX = "weakagent."


def normalize_module_name(raw: str) -> str:
    """Map user-facing module paths to importable dotted names under ``weakagent``.

    Accepts forms such as:
    - ``weakagent.tools.memory.long``
    - ``tools.memory.long``
    - ``tools/memory/long.py``
    """
    name = (raw or "").strip().replace("\\", "/").replace("/", ".")
    if name.endswith(".py"):
        name = name[:-3]
    if name.startswith("."):
        name = name.lstrip(".")
    if not name.startswith(_ALLOWED_PREFIX):
        name = f"{_ALLOWED_PREFIX}{name}"
    return name


def reload_modules(module_names: Iterable[str]) -> List[Dict[str, Any]]:
    """Reload modules already loaded (or import then reload).

    Only modules under ``weakagent.*`` are allowed.

    Returns:
        Per-module result dicts with ``module``, ``status`` (``reloaded`` | ``error``),
        and optional ``file`` / ``error`` fields.
    """
    results: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for raw in module_names:
        full_name = normalize_module_name(raw)
        if full_name in seen:
            continue
        seen.add(full_name)

        if not full_name.startswith(_ALLOWED_PREFIX):
            results.append(
                {
                    "module": full_name,
                    "status": "error",
                    "error": f"Only modules under {_ALLOWED_PREFIX!r} can be reloaded",
                }
            )
            continue

        try:
            if full_name not in sys.modules:
                importlib.import_module(full_name)
            mod = importlib.reload(sys.modules[full_name])
            results.append(
                {
                    "module": full_name,
                    "status": "reloaded",
                    "file": getattr(mod, "__file__", None),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "module": full_name,
                    "status": "error",
                    "error": str(exc),
                }
            )

    return results


def reload_weakagent_tool_modules(
    *,
    extra_modules: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Reload common tool-related modules after on-disk edits.

    Always refreshes the built-in tool registry scan after reload attempts.
    """
    from weakagent.tools.tool_collection import get_builtin_tool_registry

    defaults = [
        "weakagent.tools.memory.long",
        "weakagent.tools.sub_agent.create_sub_agent",
        "weakagent.tools.tool.hot_reload",
    ]
    names = list(defaults)
    if extra_modules:
        names.extend(extra_modules)

    results = reload_modules(names)
    get_builtin_tool_registry(refresh=True)
    return results
