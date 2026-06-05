from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from weakagent.config.settings import config
from weakagent.llm.llm import LLM
from weakagent.llm.summarize import extract_long_memory
from weakagent.memory.base import BaseMemory, MemoryType
from weakagent.memory.session import SessionMemory
from weakagent.schemas.message import Message
from weakagent.utils.logger import get_logger

logger = get_logger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LongMemoryEntry(BaseModel):
    """One persisted long-term memory item."""

    memory_id: str
    memory_type: str = "general"
    importance: float = 0.5
    content: str
    user_id: Optional[str] = None
    source_message: Optional[str] = None
    created_at: Optional[str] = None


class LongMemory(BaseMemory):
    """Long-term user memory backed by sqlite."""

    memory_type: MemoryType = Field(default=MemoryType.LONG)
    db_path: str = Field(default="weakagent.sqlite3")
    user_id: Optional[str] = None
    entries: List[LongMemoryEntry] = Field(default_factory=list)
    dedupe: bool = Field(default=True, description="Skip insert when content already exists for user_id")

    @property
    def message(self) -> Message:
        """Long-term memory message"""
        return Message.system_message(self.to_system_context())

    @model_validator(mode="after")
    def _init(self) -> "LongMemory":
        self.db_path = str(self._resolve_db_path(self.db_path))
        self._init_db()
        if self.user_id:
            self.load_for_user(self.user_id)
        return self

    @staticmethod
    def _resolve_db_path(db_path: str) -> Path:
        return config.resolve_db_path(
            db_path,
            sections=("memory", "conversation", "session", "scheduler"),
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
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT UNIQUE NOT NULL,
                    user_id TEXT,
                    memory_type TEXT,
                    importance REAL,
                    content TEXT NOT NULL,
                    source_message TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_long_term_memory_user_id ON long_term_memory(user_id)"
            )
            conn.commit()

    def _row_to_entry(self, row: sqlite3.Row) -> LongMemoryEntry:
        return LongMemoryEntry(
            memory_id=str(row["memory_id"]),
            user_id=row["user_id"],
            memory_type=str(row["memory_type"] or "general"),
            importance=float(row["importance"] or 0.5),
            content=str(row["content"] or ""),
            source_message=row["source_message"],
            created_at=row["created_at"],
        )


    #--------------------------------------------
    # Long Memory Extraction and Saving
    #--------------------------------------------
    async def extract_and_save(
        self,
        user_message: str,
        *,
        llm: Optional[LLM] = None,
    ) -> dict:
        """Run LLM extraction and persist when ``should_save`` is true."""
        llm = llm or LLM(config_name="fast")
        result = await extract_long_memory(llm, user_message)
        if result.get("should_save"):
            self.add_entry(
                content=result["memory"],
                memory_type=result.get("memory_type", "general"),
                importance=result.get("importance", 0.5),
                source_message=(user_message or "").strip()[:2000] or None,
            )
        return result

    async def extract_and_save_from_session(
        self,
        session: SessionMemory,
        *,
        llm: Optional[LLM] = None,
    ) -> dict:
        """Extract long-term memory from a runtime session transcript and persist."""
        text = session.format_for_long_memory_extraction()
        if not text or text == "[]":
            return {"should_save": False}
        return await self.extract_and_save(text, llm=llm)


    #--------------------------------------------
    # Long Memory CRUD
    #--------------------------------------------
    def load_for_user(self, user_id: Optional[str] = None) -> List[LongMemoryEntry]:
        """Load all long memories for a user into ``entries``."""
        uid = user_id if user_id is not None else self.user_id
        with self._connect() as conn:
            if uid:
                rows = conn.execute(
                    """
                    SELECT memory_id, user_id, memory_type, importance, content,
                           source_message, created_at
                    FROM long_term_memory
                    WHERE user_id = ?
                    ORDER BY importance DESC, id ASC
                    """,
                    (uid,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT memory_id, user_id, memory_type, importance, content,
                           source_message, created_at
                    FROM long_term_memory
                    ORDER BY importance DESC, id ASC
                    """
                ).fetchall()
        self.entries = [self._row_to_entry(r) for r in rows]
        return list(self.entries)

    def add_entry(
        self,
        *,
        content: str,
        memory_type: str = "general",
        importance: float = 0.5,
        user_id: Optional[str] = None,
        source_message: Optional[str] = None,
    ) -> Optional[LongMemoryEntry]:
        """Persist one memory item; returns None if deduped or content empty."""
        text = (content or "").strip()
        if not text:
            return None

        uid = user_id if user_id is not None else self.user_id
        if self.dedupe and uid:
            for e in self.entries:
                if e.user_id == uid and e.content == text:
                    logger.info("Long memory deduped for user_id=%s", uid)
                    return None

        importance = max(0.0, min(1.0, float(importance)))
        memory_id = f"ltm_{uuid.uuid4().hex[:16]}"
        created_at = _utc_now_iso()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO long_term_memory(
                    memory_id, user_id, memory_type, importance, content,
                    source_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    uid,
                    memory_type,
                    importance,
                    text,
                    source_message,
                    created_at,
                ),
            )
            conn.commit()

        entry = LongMemoryEntry(
            memory_id=memory_id,
            user_id=uid,
            memory_type=memory_type,
            importance=importance,
            content=text,
            source_message=source_message,
            created_at=created_at,
        )
        self.entries.append(entry)
        return entry

    def to_system_context(self, *, max_items: int = 20) -> str:
        """Format stored memories for injection into the system prompt."""
        if not self.entries:
            return ""
        items = sorted(self.entries, key=lambda e: e.importance, reverse=True)[:max_items]
        lines = ["[Long-term memory]"]
        for e in items:
            lines.append(f"- ({e.memory_type}, {e.importance:.2f}) {e.content}")
        return "\n".join(lines)

    def to_system_message(self) -> Message:
        """Format stored memories for injection into the system prompt."""
        return Message.system_message(self.to_system_context())

    def to_dict_list(self) -> List[Dict[str, Any]]:
        return [e.model_dump() for e in self.entries]
