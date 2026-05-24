from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from weakagent.llm.llm import LLM
from weakagent.llm.summarize import summarize_working_memory
from weakagent.memory.base import BaseMemory, MemoryType
from weakagent.schemas.message import Message
from weakagent.utils.logger import get_logger

logger = get_logger(__name__)


class WorkingMemorySummaryEntry(BaseModel):
    """One persisted working-memory skill/workflow extraction."""

    summary_id: str
    run_id: str
    source_session_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_type: Optional[str] = None
    agent_id: Optional[str] = None
    status: str = "completed"
    summary: str
    messages_count: int = 0
    extra: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None


class WorkingMemory(BaseMemory):
    """Single-run in-memory context; summaries persist to sqlite."""

    memory_type: MemoryType = Field(default=MemoryType.WORKING)
    db_path: str = Field(default="weakagent.sqlite3")

    run_id: str = Field(default_factory=lambda: f"wrun_{uuid.uuid4().hex[:12]}")
    source_session_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_type: Optional[str] = None
    agent_id: Optional[str] = None
    status: str = Field(default="completed")

    def _init_db(self) -> None:
        self.ensure_db_parent()
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS working_memory_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary_id TEXT UNIQUE NOT NULL,
                    run_id TEXT NOT NULL,
                    source_session_id TEXT,
                    user_id TEXT,
                    agent_type TEXT,
                    agent_id TEXT,
                    status TEXT DEFAULT 'completed',
                    summary TEXT NOT NULL,
                    messages_count INTEGER DEFAULT 0,
                    extra TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(run_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_working_memory_summary_run_id "
                "ON working_memory_summary(run_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_working_memory_summary_source_session_id "
                "ON working_memory_summary(source_session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_working_memory_summary_created_at "
                "ON working_memory_summary(created_at)"
            )
            conn.commit()

    async def summarize(self, *, llm: Optional[LLM] = None) -> Message:
        """Summarize current in-memory messages via LLM."""
        llm = llm or LLM(config_name="fast")
        return await summarize_working_memory(llm, self.messages)

    async def summarize_and_save(
        self,
        *,
        llm: Optional[LLM] = None,
        run_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Summarize current messages and persist to `working_memory_summary`."""
        llm = llm or LLM(config_name="fast")
        summary_msg = await self.summarize(llm=llm)
        return self.save_summary(
            summary_msg.content or "",
            run_id=run_id,
            llm=llm,
            extra=extra,
        )

    def save_summary(
        self,
        summary_text: str,
        *,
        run_id: Optional[str] = None,
        llm: Optional[LLM] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist a working-memory summary without calling the LLM."""
        text = (summary_text or "").strip()
        if not text:
            logger.warning("Skip working memory summary save: empty summary")
            return ""

        rid = run_id or self.run_id
        summary_id = f"wms_{uuid.uuid4().hex[:16]}"
        payload = dict(extra or {})
        if llm is not None:
            payload.setdefault("model", getattr(llm, "model", None))
        payload.setdefault("messages_count", len(self.messages))

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO working_memory_summary(
                    summary_id, run_id, source_session_id, user_id, agent_type,
                    agent_id, status, summary, messages_count, extra, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    summary_id = excluded.summary_id,
                    source_session_id = excluded.source_session_id,
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
                    rid,
                    self.source_session_id,
                    self.user_id,
                    self.agent_type,
                    self.agent_id,
                    self.status,
                    text,
                    len(self.messages),
                    json.dumps(payload, ensure_ascii=False),
                    self.utc_now_iso(),
                ),
            )
            conn.commit()

        logger.info(
            "Working memory summary saved run_id=%s messages=%s",
            rid,
            len(self.messages),
        )
        return text

    @classmethod
    def _row_to_entry(cls, row: Any) -> WorkingMemorySummaryEntry:
        try:
            extra = json.loads(row["extra"] or "{}")
        except Exception:
            extra = {}
        if not isinstance(extra, dict):
            extra = {}
        return WorkingMemorySummaryEntry(
            summary_id=str(row["summary_id"]),
            run_id=str(row["run_id"]),
            source_session_id=row["source_session_id"],
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
        run_id: str,
        *,
        db_path: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Optional[WorkingMemorySummaryEntry]:
        path = str(cls._resolve_db_path(db_path or "weakagent.sqlite3", MemoryType.WORKING))
        query = """
            SELECT summary_id, run_id, source_session_id, user_id, agent_type,
                   agent_id, status, summary, messages_count, extra, created_at
            FROM working_memory_summary
            WHERE run_id = ?
        """
        params: List[Any] = [run_id]
        if agent_id is not None:
            query += " AND agent_id = ?"
            params.append(agent_id)
        query += " ORDER BY id DESC LIMIT 1"

        with cls._connect_db(path) as conn:
            row = conn.execute(query, params).fetchone()
        return cls._row_to_entry(row) if row else None

    @classmethod
    def list_summaries(
        cls,
        *,
        db_path: Optional[str] = None,
        source_session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[WorkingMemorySummaryEntry]:
        path = str(cls._resolve_db_path(db_path or "weakagent.sqlite3", MemoryType.WORKING))
        query = """
            SELECT summary_id, run_id, source_session_id, user_id, agent_type,
                   agent_id, status, summary, messages_count, extra, created_at
            FROM working_memory_summary
            WHERE 1=1
        """
        params: List[Any] = []
        if source_session_id:
            query += " AND source_session_id = ?"
            params.append(source_session_id)
        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(max(1, int(limit)))

        with cls._connect_db(path) as conn:
            rows = conn.execute(query, params).fetchall()
        return [cls._row_to_entry(r) for r in rows]

    @classmethod
    def _connect_db(cls, db_path: str):
        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_latest_summary(self) -> Optional[WorkingMemorySummaryEntry]:
        return self.fetch_summary(self.run_id, db_path=self.db_path, agent_id=self.agent_id)
