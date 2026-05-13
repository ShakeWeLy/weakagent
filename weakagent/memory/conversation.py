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
from weakagent.llm.summarize import summarize_working_memory
from weakagent.memory.base import BaseMemory, MemoryType
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


class ConversationMemory(BaseMemory):
    """Persist conversation messages to sqlite tables."""

    memory_type: MemoryType = Field(default=MemoryType.SHORT)
    db_path: str = Field(default="conversation.sqlite3")

    session_id: str = Field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:16]}")
    user_id: Optional[str] = None
    agent_type: Optional[str] = None
    title: Optional[str] = None
    status: str = Field(default="active")

    @model_validator(mode="after")
    def _init(self) -> "ConversationMemory":
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
        configured = (raw.get("conversation") or {}).get("db_path")
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
                CREATE TABLE IF NOT EXISTS conversation_session (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    user_id TEXT,
                    agent_type TEXT,
                    title TEXT,
                    status TEXT DEFAULT 'active',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_message (
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
                "CREATE INDEX IF NOT EXISTS idx_conversation_message_session_id ON conversation_message(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversation_message_role ON conversation_message(role)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_call (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_call_id TEXT UNIQUE,
                    message_id TEXT,
                    tool_name TEXT,
                    tool_input TEXT,
                    tool_output TEXT,
                    status TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tool_call_message_id ON tool_call(message_id)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_summary (
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
                "CREATE INDEX IF NOT EXISTS idx_session_summary_session_id ON session_summary(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_summary_created_at ON session_summary(created_at)"
            )
            conn.commit()

    def ensure_session(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_session(
                    session_id, user_id, agent_type, title, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    agent_type = excluded.agent_type,
                    title = excluded.title,
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (
                    self.session_id,
                    self.user_id,
                    self.agent_type,
                    self.title,
                    self.status,
                    _utc_now_iso(),
                    _utc_now_iso(),
                ),
            )
            conn.commit()

    def add_message(self, message: Message, extra: Optional[Dict[str, Any]] = None) -> None:
        super().add_message(message)
        try:
            self.ensure_session()
            self._persist_message(message, extra=extra)
        except Exception:
            logger.exception("Failed to persist conversation message")

    def list_session_messages(self, *, session_id: Optional[str] = None) -> List[Message]:
        sid = session_id or self.session_id
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, extra
                FROM conversation_message
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (sid,),
            ).fetchall()

        msgs: List[Message] = []
        for r in rows:
            role = str(r["role"])
            content = r["content"]
            name = None
            tool_call_id = None
            try:
                extra = json.loads(r["extra"] or "{}")
                name = extra.get("name")
                tool_call_id = extra.get("tool_call_id")
            except Exception:
                pass
            msgs.append(
                Message(
                    role=role,  # type: ignore[arg-type]
                    content=content,
                    name=name,
                    tool_call_id=tool_call_id,
                )
            )
        return msgs

    async def write_session_summary(
        self,
        *,
        run_id: str,
        status: str,
        llm: Optional[LLM] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Summarize current session and persist to `session_summary`."""
        llm = llm or LLM(config_name="fast")
        messages = self.list_session_messages()
        summary_msg = await summarize_working_memory(llm, messages)
        summary_text = summary_msg.content or ""

        payload = extra or {}
        payload.setdefault("model", getattr(llm, "model", None))
        payload.setdefault("messages_count", len(messages))

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_summary(
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

    def _persist_message(self, message: Message, extra: Optional[Dict[str, Any]] = None) -> None:
        message_id = f"msg_{uuid.uuid4().hex[:16]}"
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
                INSERT INTO conversation_message(
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
                "UPDATE conversation_session SET updated_at = ? WHERE session_id = ?",
                (_utc_now_iso(), self.session_id),
            )

            if message.tool_calls:
                for call in message.tool_calls:
                    call_dict = _to_dict(call)
                    tc_id = call_dict.get("id")
                    fn = call_dict.get("function") or {}
                    tool_name = fn.get("name")
                    tool_input = fn.get("arguments")
                    conn.execute(
                        """
                        INSERT INTO tool_call(
                            tool_call_id, message_id, tool_name, tool_input, status, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(tool_call_id) DO UPDATE SET
                            message_id = excluded.message_id,
                            tool_name = excluded.tool_name,
                            tool_input = excluded.tool_input,
                            status = excluded.status
                        """,
                        (
                            tc_id,
                            message_id,
                            tool_name,
                            json.dumps(tool_input, ensure_ascii=False)
                            if isinstance(tool_input, (dict, list))
                            else str(tool_input) if tool_input is not None else None,
                            "running",
                            _utc_now_iso(),
                        ),
                    )

            if message.role == "tool" and message.tool_call_id:
                tool_status = "failed" if (message.content or "").startswith("Error") else "success"
                updated = conn.execute(
                    """
                    UPDATE tool_call
                    SET tool_output = ?, status = ?
                    WHERE tool_call_id = ?
                    """,
                    (message.content, tool_status, message.tool_call_id),
                )
                if updated.rowcount == 0:
                    conn.execute(
                        """
                        INSERT INTO tool_call(
                            tool_call_id, message_id, tool_name, tool_output, status, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            message.tool_call_id,
                            message_id,
                            message.name,
                            message.content,
                            tool_status,
                            _utc_now_iso(),
                        ),
                    )
            conn.commit()
