from __future__ import annotations

import asyncio
import json
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from weakagent.config.settings import config
from weakagent.utils.logger import get_logger

logger = get_logger(__name__)

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(int, Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _dt_to_ts(dt: Optional[datetime]) -> Optional[float]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        # Treat naive datetimes as UTC for persistence consistency.
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _ts_to_dt(ts: Optional[float]) -> Optional[datetime]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc)


class Task(BaseModel):
    """Persistable task model (data only; execution happens in Executors)."""

    id: int = Field(..., description="Task id (primary key)")
    task_type: str = Field(..., description="Task type used for routing")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM)

    payload: Dict[str, Any] = Field(default_factory=dict, description="Task params/payload")

    next_run_at: Optional[datetime] = Field(default=None, description="When task becomes due")
    attempts: int = Field(default=0, ge=0)
    last_error: Optional[str] = None

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class Executor(ABC):
    """真正干活的执行器。"""

    @abstractmethod
    async def execute(self, task: Task, store: TaskStore) -> None:  # noqa: F821
        raise NotImplementedError


class TaskRegistry:
    """运行时注册中心：task_type -> executor class."""

    def __init__(self) -> None:
        self._map: Dict[str, Type[Executor]] = {}

    def register(self, task_type: str, executor_cls: Type[Executor]) -> None:
        if not task_type:
            raise ValueError("task_type cannot be empty")
        if not issubclass(executor_cls, Executor):
            raise TypeError("executor_cls must be a subclass of Executor")
        self._map[task_type] = executor_cls

    def unregister(self, task_type: str) -> bool:
        return self._map.pop(task_type, None) is not None

    def get(self, task_type: str) -> Type[Executor]:
        try:
            return self._map[task_type]
        except KeyError as e:
            raise KeyError(f"Unknown task_type: {task_type}") from e

    @property
    def supported_types(self) -> List[str]:
        return sorted(self._map.keys())


@dataclass(frozen=True)
class TaskRow:
    id: int
    task_type: str
    status: str
    priority: int
    payload_json: str
    next_run_at_ts: Optional[float]
    attempts: int
    last_error: Optional[str]
    created_at_ts: Optional[float]
    updated_at_ts: Optional[float]


class TaskStore:
    """SQLite 存储层：只做数据存取（CRUD + 状态更新）。"""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = str(self._resolve_db_path(db_path))
        self._init_db()

    @staticmethod
    def _resolve_db_path(db_path: str | Path | None) -> Path:
        if db_path is not None:
            return Path(db_path)
        return config.resolve_db_path(
            config.scheduler.db_path,
            sections=("scheduler",),
        )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                  id INTEGER PRIMARY KEY,
                  task_type TEXT NOT NULL,
                  status TEXT NOT NULL,
                  priority INTEGER NOT NULL,
                  payload_json TEXT NOT NULL,
                  next_run_at_ts REAL,
                  attempts INTEGER NOT NULL DEFAULT 0,
                  last_error TEXT,
                  created_at_ts REAL,
                  updated_at_ts REAL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_due ON tasks(status, next_run_at_ts, priority)"
            )
            conn.commit()

    def create_task(
        self,
        task_type: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        next_run_at: Optional[datetime] = None,
        task_id: Optional[int] = None,
    ) -> Task:
        now = _utc_now()
        payload_json = json.dumps(payload or {}, ensure_ascii=False)
        with self._connect() as conn:
            if task_id is None:
                cur = conn.execute(
                    """
                    INSERT INTO tasks(task_type, status, priority, payload_json, next_run_at_ts, attempts, last_error, created_at_ts, updated_at_ts)
                    VALUES (?, ?, ?, ?, ?, 0, NULL, ?, ?)
                    """,
                    (
                        task_type,
                        TaskStatus.PENDING.value,
                        int(priority),
                        payload_json,
                        _dt_to_ts(next_run_at),
                        _dt_to_ts(now),
                        _dt_to_ts(now),
                    ),
                )
                new_id = int(cur.lastrowid)
            else:
                conn.execute(
                    """
                    INSERT INTO tasks(id, task_type, status, priority, payload_json, next_run_at_ts, attempts, last_error, created_at_ts, updated_at_ts)
                    VALUES (?, ?, ?, ?, ?, ?, 0, NULL, ?, ?)
                    """,
                    (
                        int(task_id),
                        task_type,
                        TaskStatus.PENDING.value,
                        int(priority),
                        payload_json,
                        _dt_to_ts(next_run_at),
                        _dt_to_ts(now),
                        _dt_to_ts(now),
                    ),
                )
                new_id = int(task_id)
            conn.commit()
        return self.get_task(new_id)

    def get_task(self, task_id: int) -> Task:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (int(task_id),)).fetchone()
            if row is None:
                raise KeyError(f"task_id not found: {task_id}")
            return self._row_to_task(row)

    def update_task(
        self,
        task_id: int,
        *,
        payload: Optional[Dict[str, Any]] = None,
        priority: Optional[TaskPriority] = None,
        next_run_at: Optional[datetime] = None,
    ) -> Task:
        now = _utc_now()
        updates: List[str] = []
        params: List[Any] = []
        if payload is not None:
            updates.append("payload_json = ?")
            params.append(json.dumps(payload, ensure_ascii=False))
        if priority is not None:
            updates.append("priority = ?")
            params.append(int(priority))
        if next_run_at is not None:
            updates.append("next_run_at_ts = ?")
            params.append(_dt_to_ts(next_run_at))
        updates.append("updated_at_ts = ?")
        params.append(_dt_to_ts(now))

        if not updates:
            return self.get_task(task_id)

        with self._connect() as conn:
            conn.execute(
                f"UPDATE tasks SET {', '.join(updates)} WHERE id = ?",
                (*params, int(task_id)),
            )
            conn.commit()
        return self.get_task(task_id)

    def list_tasks(self, *, status: Optional[TaskStatus] = None) -> List[Task]:
        with self._connect() as conn:
            if status is None:
                rows = conn.execute("SELECT * FROM tasks ORDER BY id DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM tasks WHERE status = ? ORDER BY id DESC",
                    (status.value,),
                ).fetchall()
            return [self._row_to_task(r) for r in rows]

    def get_due_tasks(self, now: Optional[datetime] = None, *, limit: int = 50) -> List[Task]:
        now = now or _utc_now()
        now_ts = _dt_to_ts(now)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM tasks
                WHERE status = ?
                  AND (next_run_at_ts IS NULL OR next_run_at_ts <= ?)
                ORDER BY priority DESC, COALESCE(next_run_at_ts, 0) ASC, id ASC
                LIMIT ?
                """,
                (TaskStatus.PENDING.value, now_ts, int(limit)),
            ).fetchall()
            return [self._row_to_task(r) for r in rows]

    def mark_running(self, task_id: int) -> None:
        self._set_status(task_id, TaskStatus.RUNNING, last_error=None, bump_attempts=True)

    def mark_completed(self, task_id: int) -> None:
        self._set_status(task_id, TaskStatus.COMPLETED, last_error=None, bump_attempts=False)

    def mark_failed(self, task_id: int, *, error: str) -> None:
        self._set_status(task_id, TaskStatus.FAILED, last_error=error, bump_attempts=False)

    def delete_task(self, task_id: int) -> bool:
        """Remove a task row permanently."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM tasks WHERE id = ?", (int(task_id),))
            conn.commit()
            return int(cur.rowcount) > 0

    def cancel_task(self, task_id: int) -> Task:
        """Mark a task as cancelled without deleting it."""
        self._set_status(task_id, TaskStatus.CANCELLED, last_error=None, bump_attempts=False)
        return self.get_task(task_id)

    def mark_pending(self, task_id: int, *, next_run_at: Optional[datetime] = None) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET status = ?, next_run_at_ts = ?, updated_at_ts = ?
                WHERE id = ?
                """,
                (
                    TaskStatus.PENDING.value,
                    _dt_to_ts(next_run_at),
                    _dt_to_ts(now),
                    int(task_id),
                ),
            )
            conn.commit()

    def _set_status(
        self,
        task_id: int,
        status: TaskStatus,
        *,
        last_error: Optional[str],
        bump_attempts: bool,
    ) -> None:
        now = _utc_now()
        with self._connect() as conn:
            if bump_attempts:
                conn.execute(
                    """
                    UPDATE tasks
                    SET status = ?, attempts = attempts + 1, last_error = ?, updated_at_ts = ?
                    WHERE id = ?
                    """,
                    (status.value, last_error, _dt_to_ts(now), int(task_id)),
                )
            else:
                conn.execute(
                    """
                    UPDATE tasks
                    SET status = ?, last_error = ?, updated_at_ts = ?
                    WHERE id = ?
                    """,
                    (status.value, last_error, _dt_to_ts(now), int(task_id)),
                )
            conn.commit()

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> Task:
        payload = json.loads(row["payload_json"] or "{}")
        return Task(
            id=int(row["id"]),
            task_type=str(row["task_type"]),
            status=TaskStatus(str(row["status"])),
            priority=TaskPriority(int(row["priority"])),
            payload=payload if isinstance(payload, dict) else {},
            next_run_at=_ts_to_dt(row["next_run_at_ts"]),
            attempts=int(row["attempts"]),
            last_error=row["last_error"],
            created_at=_ts_to_dt(row["created_at_ts"]),
            updated_at=_ts_to_dt(row["updated_at_ts"]),
        )


class Dispatcher:
    """任务 -> 对应执行器 -> 执行。"""

    def __init__(self, *, registry: TaskRegistry, store: TaskStore) -> None:
        self.registry = registry
        self.store = store

    async def dispatch(self, task: Task) -> None:
        executor_cls = self.registry.get(task.task_type)
        executor = executor_cls()

        self.store.mark_running(task.id)
        try:
            await executor.execute(task, self.store)
        except asyncio.CancelledError as exc:
            logger.warning(
                "Task execution cancelled. task_id=%s type=%s",
                task.id,
                task.task_type,
            )
            self.store.mark_failed(task.id, error=f"cancelled: {exc}")
            return
        except Exception as exc:
            logger.exception("Task execution failed. task_id=%s type=%s", task.id, task.task_type)
            self.store.mark_failed(task.id, error=str(exc))
            return

        # Executor is responsible for updating task next_run_at/retry if needed.
        # If it didn't set a terminal status, we mark completed by default.
        refreshed = self.store.get_task(task.id)
        if refreshed.status == TaskStatus.RUNNING:
            self.store.mark_completed(task.id)


class Scheduler:
    """时间扫描器：只负责找 due tasks 并交给 dispatcher。"""

    def __init__(self, *, store: TaskStore, dispatcher: Dispatcher) -> None:
        self.store = store
        self.dispatcher = dispatcher

    async def run_once(self, *, now: Optional[datetime] = None, limit: int = 50) -> int:
        due = self.store.get_due_tasks(now=now, limit=limit)
        for task in due:
            await self.dispatcher.dispatch(task)
        return len(due)
