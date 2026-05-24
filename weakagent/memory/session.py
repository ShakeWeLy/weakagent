from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from weakagent.llm.llm import LLM
from weakagent.llm.summarize import generate_session_title, summarize_short_memory
from weakagent.memory.base import BaseMemory, MemoryType
from weakagent.schemas.message import Message
from weakagent.utils.logger import get_logger

logger = get_logger(__name__)


class SessionMemorySummaryEntry(BaseModel):
    """Runtime-level session summary (one per interactive runtime loop)."""

    summary_id: str
    session_id: str
    user_id: Optional[str] = None
    agent_type: Optional[str] = None
    agent_id: Optional[str] = None
    status: str = "completed"
    summary: str
    messages_count: int = 0
    extra: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None


class SessionRecord(BaseModel):
    """Session metadata row."""

    session_id: str
    user_id: Optional[str] = None
    agent_type: Optional[str] = None
    agent_id: Optional[str] = None
    title: Optional[str] = None
    status: str = "active"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SessionMemory(BaseMemory):
    """Runtime-scoped session: metadata + summary; messages live in ConversationMemory."""

    memory_type: MemoryType = Field(default=MemoryType.SESSION)
    db_path: str = Field(default="weakagent.sqlite3")

    session_id: str = Field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:16]}")
    run_id: Optional[str] = Field(
        default=None,
        description="Current agent.run id (stamped on conversation rows via agent)",
    )
    user_id: Optional[str] = None
    agent_type: Optional[str] = None
    agent_id: Optional[str] = None
    title: Optional[str] = None
    status: str = Field(default="active")

    @model_validator(mode="after")
    def _init_session(self) -> "SessionMemory":
        self.ensure_session()
        return self

    def _init_db(self) -> None:
        self.ensure_db_parent()
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session (
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
                CREATE TABLE IF NOT EXISTS session_memory_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary_id TEXT UNIQUE NOT NULL,
                    session_id TEXT NOT NULL,
                    user_id TEXT,
                    agent_type TEXT,
                    agent_id TEXT,
                    status TEXT DEFAULT 'completed',
                    summary TEXT NOT NULL,
                    messages_count INTEGER DEFAULT 0,
                    extra TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_memory_summary_session_id "
                "ON session_memory_summary(session_id)"
            )
            self._migrate_schema(conn)
            conn.commit()

    @staticmethod
    def _migrate_schema(conn) -> None:
        """Upgrade legacy session_memory_summary (run_id UNIQUE) to session_id UNIQUE."""
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='session_memory_summary'"
        ).fetchone()
        ddl = str(row["sql"]) if row else ""
        if ddl and "UNIQUE(session_id)" not in ddl:
            conn.execute(
                "ALTER TABLE session_memory_summary "
                "RENAME TO session_memory_summary_legacy"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_memory_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary_id TEXT UNIQUE NOT NULL,
                    session_id TEXT NOT NULL,
                    user_id TEXT,
                    agent_type TEXT,
                    agent_id TEXT,
                    status TEXT DEFAULT 'completed',
                    summary TEXT NOT NULL,
                    messages_count INTEGER DEFAULT 0,
                    extra TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_memory_summary_session_id "
                "ON session_memory_summary(session_id)"
            )

    def ensure_session(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session(
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
                    self.utc_now_iso(),
                    self.utc_now_iso(),
                ),
            )
            conn.commit()

    async def generate_title_from_request(
        self,
        request: str,
        *,
        llm: Optional[LLM] = None,
    ) -> Optional[str]:
        text = (request or "").strip()
        if not text:
            return self.title
        if self.reload_messages():
            return self.title

        llm = llm or LLM(config_name="fast")
        try:
            title = await generate_session_title(llm, text)
        except Exception:
            logger.exception("LLM session title generation failed")
            title = ""

        if not title:
            title = text[:50] + ("..." if len(text) > 50 else "")

        self.title = title.strip()[:200]
        self.ensure_session()
        return self.title

    @classmethod
    def get_last_session_id(cls, db_path: Optional[str] = None) -> Optional[str]:
        path = str(cls._resolve_db_path(db_path or "weakagent.sqlite3", MemoryType.SESSION))
        with cls._connect_db(path) as conn:
            row = conn.execute(
                """
                SELECT session_id FROM session
                ORDER BY updated_at DESC, id DESC
                LIMIT 1
                """
            ).fetchone()
        return str(row["session_id"]) if row else None

    @classmethod
    def get_last_session_id_for_agent(
        cls,
        agent_id: str,
        *,
        db_path: Optional[str] = None,
        exclude_session_id: Optional[str] = None,
    ) -> Optional[str]:
        """Most recently updated session row for ``agent_id`` (optional exclude current)."""
        path = str(cls._resolve_db_path(db_path or "weakagent.sqlite3", MemoryType.SESSION))
        query = """
            SELECT session_id FROM session
            WHERE agent_id = ?
        """
        params: List[Any] = [agent_id]
        if exclude_session_id:
            query += " AND session_id != ?"
            params.append(exclude_session_id)
        query += " ORDER BY updated_at DESC, id DESC LIMIT 1"
        with cls._connect_db(path) as conn:
            row = conn.execute(query, params).fetchone()
        return str(row["session_id"]) if row else None

    def reload_messages(self) -> List[Message]:
        """Reload full runtime transcript from ConversationMemory."""
        from weakagent.memory.conversation import ConversationMemory

        loaded = ConversationMemory.fetch_messages(
            db_path=self.db_path,
            session_id=self.session_id,
            agent_id=self.agent_id,
        )
        self.messages = loaded
        return list(self.messages)

    async def summarize(self, *, llm: Optional[LLM] = None) -> Message:
        llm = llm or LLM(config_name="fast")
        msgs = self.messages or self.reload_messages()
        return await summarize_short_memory(llm, msgs)

    async def finalize_runtime_summary(
        self,
        *,
        llm: Optional[LLM] = None,
        status: str = "closed",
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Summarize the full runtime session and persist (call when runtime loop ends)."""
        self.status = status
        self.ensure_session()
        self.reload_messages()

        if not self.messages:
            logger.info("Skip session summary: no messages session_id=%s", self.session_id)
            return ""

        if not self.title:
            first_user = next((m for m in self.messages if m.role == "user" and m.content), None)
            if first_user:
                await self.generate_title_from_request(first_user.content or "", llm=llm)

        llm = llm or LLM(config_name="fast")
        summary_msg = await self.summarize(llm=llm)
        return self.save_summary(
            summary_msg.content or "",
            status=status,
            llm=llm,
            extra=extra,
        )

    def save_summary(
        self,
        summary_text: str,
        *,
        status: Optional[str] = None,
        llm: Optional[LLM] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        text = (summary_text or "").strip()
        if not text:
            logger.warning("Skip session summary save: empty summary")
            return ""

        summary_id = f"sessms_{uuid.uuid4().hex[:16]}"
        payload = dict(extra or {})
        if llm is not None:
            payload.setdefault("model", getattr(llm, "model", None))
        payload.setdefault("messages_count", len(self.messages))

        st = status or self.status
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_memory_summary(
                    summary_id, session_id, user_id, agent_type,
                    agent_id, status, summary, messages_count, extra, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary_id = excluded.summary_id,
                    user_id = excluded.user_id,
                    agent_type = excluded.agent_type,
                    agent_id = excluded.agent_id,
                    status = excluded.status,
                    summary = excluded.summary,
                    messages_count = excluded.messages_count,
                    extra = excluded.extra,
                    created_at = excluded.created_at
                """,
                (
                    summary_id,
                    self.session_id,
                    self.user_id,
                    self.agent_type,
                    self.agent_id,
                    st,
                    text,
                    len(self.messages),
                    json.dumps(payload, ensure_ascii=False),
                    self.utc_now_iso(),
                ),
            )
            conn.execute(
                "UPDATE session SET status = ?, updated_at = ? WHERE session_id = ?",
                (st, self.utc_now_iso(), self.session_id),
            )
            conn.commit()

        logger.info(
            "Session runtime summary saved session_id=%s messages=%s",
            self.session_id,
            len(self.messages),
        )
        return text

    @classmethod
    def _row_to_summary_entry(cls, row: Any) -> SessionMemorySummaryEntry:
        try:
            extra = json.loads(row["extra"] or "{}")
        except Exception:
            extra = {}
        if not isinstance(extra, dict):
            extra = {}
        return SessionMemorySummaryEntry(
            summary_id=str(row["summary_id"]),
            session_id=str(row["session_id"]),
            user_id=row["user_id"],
            agent_type=row["agent_type"],
            agent_id=row["agent_id"],
            status=str(row["status"] or "completed"),
            summary=str(row["summary"] or ""),
            messages_count=int(row["messages_count"] or 0),
            extra=extra,
            created_at=row["created_at"],
        )

    @classmethod
    def fetch_summary(
        cls,
        session_id: str,
        *,
        db_path: Optional[str] = None,
    ) -> Optional[SessionMemorySummaryEntry]:
        path = str(cls._resolve_db_path(db_path or "weakagent.sqlite3", MemoryType.SESSION))
        with cls._connect_db(path) as conn:
            row = conn.execute(
                """
                SELECT summary_id, session_id, user_id, agent_type,
                       agent_id, status, summary, messages_count, extra, created_at
                FROM session_memory_summary
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        return cls._row_to_summary_entry(row) if row else None

    @classmethod
    def list_summaries(
        cls,
        *,
        db_path: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[SessionMemorySummaryEntry]:
        path = str(cls._resolve_db_path(db_path or "weakagent.sqlite3", MemoryType.SESSION))
        query = """
            SELECT summary_id, session_id, user_id, agent_type,
                   agent_id, status, summary, messages_count, extra, created_at
            FROM session_memory_summary
            WHERE 1=1
        """
        params: List[Any] = []
        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(max(1, int(limit)))

        with cls._connect_db(path) as conn:
            rows = conn.execute(query, params).fetchall()
        return [cls._row_to_summary_entry(r) for r in rows]

    @classmethod
    def _connect_db(cls, db_path: str):
        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_latest_summary(self) -> Optional[SessionMemorySummaryEntry]:
        return self.fetch_summary(self.session_id, db_path=self.db_path)

    def format_for_long_memory_extraction(self) -> str:
        """Serialize session transcript for long-memory LLM extraction."""
        msgs = self.messages or self.reload_messages()
        payload = [
            {"role": str(m.role), "content": m.content or ""}
            for m in msgs
        ]
        return json.dumps(payload, ensure_ascii=False)
