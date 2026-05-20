from __future__ import annotations

import json
import sqlite3
import tomllib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, model_validator

from weakagent.config.settings import PROJECT_ROOT
from weakagent.llm.llm import LLM
from weakagent.llm.summarize import generate_session_title, summarize_working_memory
from weakagent.memory.base import BaseMemory, MemoryType
from weakagent.memory.conversation import message_from_conversation_row
from weakagent.schemas.message import Message
from weakagent.utils.logger import get_logger

logger = get_logger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        data = dump()
        return data if isinstance(data, dict) else {}
    return {}


class RuntimeSessionStore(BaseMemory):
    """Persist runtime memory (request + last_result pairs) to sqlite."""

    memory_type: MemoryType = Field(default=MemoryType.RUNTIME)
    db_path: str = Field(default="weakagent.sqlite3")

    session_id: str = Field(default_factory=lambda: f"rsess_{uuid.uuid4().hex[:16]}")
    user_id: Optional[str] = None
    agent_type: Optional[str] = None
    agent_id: Optional[str] = None
    title: Optional[str] = None
    status: str = Field(default="active")

    @model_validator(mode="after")
    def _init(self) -> "RuntimeSessionStore":
        self.db_path = str(self._resolve_db_path(self.db_path))
        self._init_db()
        self.ensure_session()
        return self

    @staticmethod
    def _resolve_db_path(db_path: str) -> Path:
        p = Path(db_path)
        if p.is_absolute():
            return p
        cfg_path = PROJECT_ROOT / "config.toml"
        try:
            raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            raw = {}
        configured = (raw.get("runtime_session") or {}).get("db_path")
        if configured:
            cp = Path(str(configured))
            return cp if cp.is_absolute() else (PROJECT_ROOT / cp)
        return PROJECT_ROOT / p

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime_session (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    user_id TEXT,
                    agent_type TEXT,
                    agent_id TEXT,
                    title TEXT,
                    status TEXT DEFAULT 'active',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime_session_message (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT UNIQUE NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT,
                    extra TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_runtime_session_message_session_id "
                "ON runtime_session_message(session_id)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime_session_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    agent_type TEXT,
                    status TEXT,
                    summary TEXT,
                    extra TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_id, run_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_runtime_session_summary_session_id "
                "ON runtime_session_summary(session_id)"
            )
            self._migrate_schema(conn)
            conn.commit()

    @staticmethod
    def _migrate_schema(conn: sqlite3.Connection) -> None:
        """Apply lightweight schema upgrades for existing sqlite files."""
        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(runtime_session)").fetchall()
        }
        if cols and "agent_id" not in cols:
            conn.execute("ALTER TABLE runtime_session ADD COLUMN agent_id TEXT")

    def ensure_session(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runtime_session(
                    session_id, user_id, agent_type, agent_id, title, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    agent_type = excluded.agent_type,
                    agent_id = excluded.agent_id,
                    title = excluded.title,
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (
                    self.session_id,
                    self.user_id,
                    self.agent_type,
                    self.agent_id,
                    self.title,
                    self.status,
                    _utc_now_iso(),
                    _utc_now_iso(),
                ),
            )
            conn.commit()

    def _persist_message(self, message: Message, extra: Optional[Dict[str, Any]] = None) -> None:
        message_id = f"rmsg_{uuid.uuid4().hex[:16]}"
        payload: Dict[str, Any] = {
            "name": message.name,
            "tool_call_id": message.tool_call_id,
            "base64_image": bool(message.base64_image),
        }
        if message.tool_calls is not None:
            payload["tool_calls"] = [_to_dict(tc) for tc in message.tool_calls]
        if extra:
            payload.update(extra)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runtime_session_message(
                    message_id, session_id, role, content, extra, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    self.session_id,
                    str(message.role),
                    message.content,
                    json.dumps(payload, ensure_ascii=False),
                    _utc_now_iso(),
                ),
            )
            conn.execute(
                "UPDATE runtime_session SET updated_at = ? WHERE session_id = ?",
                (_utc_now_iso(), self.session_id),
            )
            conn.commit()

    @classmethod
    def get_last_session_id(cls, db_path: Optional[str] = None) -> Optional[str]:
        path = str(cls._resolve_db_path(db_path or "weakagent.sqlite3"))
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT session_id
                FROM runtime_session
                ORDER BY updated_at DESC, id DESC
                LIMIT 1
                """
            ).fetchone()
        return str(row["session_id"]) if row else None

    @classmethod
    def fetch_session_messages(
        cls,
        session_id: str,
        *,
        db_path: Optional[str] = None,
    ) -> List[Message]:
        path = str(cls._resolve_db_path(db_path or "weakagent.sqlite3"))
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT role, content, extra
                FROM runtime_session_message
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        return [message_from_conversation_row(r) for r in rows]

    def list_session_messages(self, *, session_id: Optional[str] = None) -> List[Message]:
        return self.fetch_session_messages(session_id or self.session_id, db_path=self.db_path)

    async def write_session_summary(
        self,
        *,
        run_id: str,
        status: str,
        messages: Optional[List[Message]] = None,
        llm: Optional[LLM] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Summarize runtime session messages and persist to `runtime_session_summary`."""
        llm = llm or LLM(config_name="fast")
        msgs = messages if messages is not None else self.list_session_messages()
        summary_msg = await summarize_working_memory(llm, msgs)
        summary_text = summary_msg.content or ""

        payload = extra or {}
        payload.setdefault("model", getattr(llm, "model", None))
        payload.setdefault("messages_count", len(msgs))

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runtime_session_summary(
                    session_id, run_id, agent_type, status, summary, extra, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, run_id) DO UPDATE SET
                    agent_type = excluded.agent_type,
                    status = excluded.status,
                    summary = excluded.summary,
                    extra = excluded.extra
                """,
                (
                    self.session_id,
                    run_id,
                    self.agent_type,
                    status,
                    summary_text,
                    json.dumps(payload, ensure_ascii=False),
                    _utc_now_iso(),
                ),
            )
            conn.commit()
        return summary_text

    async def generate_title_from_request(self, request: str, *, llm: Optional[LLM] = None) -> Optional[str]:
        text = (request or "").strip()
        if not text or self.title:
            return self.title
        llm = llm or LLM(config_name="fast")
        try:
            title = await generate_session_title(llm, text)
        except Exception:
            logger.exception("LLM runtime session title generation failed")
            title = ""
        if not title:
            title = text[:50] + ("..." if len(text) > 50 else "")
        self.title = title.strip()[:200]
        self.ensure_session()
        return self.title
