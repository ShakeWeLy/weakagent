from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from weakagent.scheduler import Task, TaskStore

_store: Optional[TaskStore] = None


def get_task_store() -> TaskStore:
    """Return the shared TaskStore used by scheduler tools."""
    global _store
    if _store is None:
        _store = TaskStore()
    return _store


def set_task_store(store: TaskStore) -> None:
    """Bind scheduler tools to a specific TaskStore instance."""
    global _store
    _store = store


def parse_next_run_at(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO-8601 datetime strings for next_run_at."""
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def task_to_dict(task: Task) -> Dict[str, Any]:
    """Serialize a Task for tool responses."""
    return task.model_dump(mode="json")
