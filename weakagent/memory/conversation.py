from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Iterable, List, Optional

from pydantic import Field

from weakagent.memory.base import BaseMemory, MemoryType
from weakagent.memory.message_store import (
    message_from_storage_row,
    select_last_n_messages_with_integrity,
    to_extra_dict,
)
from weakagent.memory.message import Message
from weakagent.utils.logger import get_logger

logger = get_logger(__name__)


class ConversationMemory(BaseMemory):
    """Append-only store: persist every message at the most basic level.

    No session metadata or summaries here — those live in ``SessionMemory``.
    """

    memory_type: MemoryType = Field(default=MemoryType.CONVERSATION)
    db_path: str = Field(default="weakagent.sqlite3")

    session_id: Optional[str] = Field(
        default=None,
        description="Optional link to runtime SessionMemory.session_id",
    )
    run_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_type: Optional[str] = None
    agent_id: Optional[str] = None

    def _init_db(self) -> None:
        self.ensure_db_parent()
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_message (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT UNIQUE NOT NULL,
                    session_id TEXT,
                    run_id TEXT,
                    user_id TEXT,
                    agent_type TEXT,
                    agent_id TEXT,
                    role TEXT NOT NULL,
                    content TEXT,
                    extra TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._migrate_schema(conn)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversation_message_session_id "
                "ON conversation_message(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversation_message_run_id "
                "ON conversation_message(run_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversation_message_created_at "
                "ON conversation_message(created_at)"
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
            conn.commit()

    @staticmethod
    def _migrate_schema(conn) -> None:
        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(conversation_message)").fetchall()
        }
        if not cols:
            return
        for col in ("run_id", "user_id", "agent_type", "agent_id"):
            if col not in cols:
                conn.execute(f"ALTER TABLE conversation_message ADD COLUMN {col} TEXT")

    def add_message(self, message: Message, extra: Optional[Dict[str, Any]] = None) -> None:
        super().add_message(message)
        try:
            self._persist_message(message, extra=extra)
        except Exception:
            logger.exception("Failed to persist conversation message")

    @classmethod
    def fetch_messages(
        cls,
        *,
        db_path: Optional[str] = None,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        last_n: Optional[int] = None,
        exclude_roles: Optional[Iterable[str]] = None,
    ) -> List[Message]:
        path = str(cls._resolve_db_path(db_path or "weakagent.sqlite3", MemoryType.CONVERSATION))
        query = """
            SELECT role, content, extra
            FROM conversation_message
            WHERE 1=1
        """
        params: List[Any] = []
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        query += " ORDER BY id ASC"

        with cls._connect_db(path) as conn:
            rows = conn.execute(query, params).fetchall()

        messages = [message_from_storage_row(r) for r in rows]
        if exclude_roles:
            excluded = set(exclude_roles)
            messages = [m for m in messages if m.role not in excluded]
        if last_n is not None:
            messages = select_last_n_messages_with_integrity(messages, last_n)
        return messages

    def list_messages(self) -> List[Message]:
        return self.fetch_messages(
            db_path=self.db_path,
            session_id=self.session_id,
            run_id=self.run_id,
            agent_id=self.agent_id,
        )

    @classmethod
    def _connect_db(cls, db_path: str):
        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _persist_message(self, message: Message, extra: Optional[Dict[str, Any]] = None) -> None:
        message_id = f"cmsg_{uuid.uuid4().hex[:16]}"
        payload: Dict[str, Any] = {
            "name": message.name,
            "tool_call_id": message.tool_call_id,
            "base64_image": bool(message.base64_image),
        }
        if message.tool_calls is not None:
            payload["tool_calls"] = [to_extra_dict(tc) for tc in message.tool_calls]
        if message.reasoning_content is not None:
            payload["reasoning_content"] = message.reasoning_content
        if extra:
            payload.update(extra)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_message(
                    message_id, session_id, run_id, user_id, agent_type,
                    agent_id, role, content, extra, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    self.session_id,
                    self.run_id,
                    self.user_id,
                    self.agent_type,
                    self.agent_id,
                    str(message.role),
                    message.content,
                    json.dumps(payload, ensure_ascii=False),
                    self.utc_now_iso(),
                ),
            )

            if message.tool_calls:
                for call in message.tool_calls:
                    call_dict = to_extra_dict(call)
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
                            self.utc_now_iso(),
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
                            self.utc_now_iso(),
                        ),
                    )
            conn.commit()
