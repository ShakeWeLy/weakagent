from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from weakagent.llm.llm import LLM
from weakagent.llm.summarize import summarize_working_memory
from weakagent.memory.base import BaseMemory, MemoryType
from weakagent.memory.message import Message
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
        owns_llm = llm is None
        llm = llm or LLM(config_name="fast")
        try:
            summary_msg = await self.summarize(llm=llm)
            return self.save_summary(
                summary_msg.content or "",
                run_id=run_id,
                llm=llm,
                extra=extra,
            )
        finally:
            # Close httpx before asyncio.run() tears down the loop (background thread).
            if owns_llm:
                await llm.client.close()

    def summarize_and_save_sync(
        self,
        *,
        llm: Optional[LLM] = None,
        run_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Sync entry point for background threads (wraps ``summarize_and_save``)."""
        try:
            return asyncio.run(
                self.summarize_and_save(llm=llm, run_id=run_id, extra=extra)
            )
        except Exception:
            logger.exception("Working memory summarize_and_save failed run_id=%s", run_id)
            return ""

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
    def _connect_db(cls, db_path: str):
        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
