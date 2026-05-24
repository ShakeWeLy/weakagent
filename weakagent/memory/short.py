from __future__ import annotations

import json
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from weakagent.llm.summarize import summarize_short_memory
from weakagent.memory.base import BaseMemory, MemoryType
from weakagent.schemas.message import Message
from weakagent.utils.logger import get_logger

logger = get_logger(__name__)


class MemoryCleanupStrategy(str, Enum):
    KEEP_LAST_N = "keep_last_n"
    TRUNCATE_TOOL_OUTPUT = "truncate_tool_output"
    SUMMARIZE_THEN_KEEP_LAST_N = "summarize_then_keep_last_n"


class ShortMemorySnapshotEntry(BaseModel):
    """One persisted full short-memory message snapshot."""

    snapshot_id: str
    run_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_type: Optional[str] = None
    agent_id: Optional[str] = None
    status: str = "completed"
    messages_count: int = 0
    extra: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None


class ShortMemory(BaseMemory):
    """Per-run agent context (in-memory); full snapshots persist to sqlite."""

    memory_type: MemoryType = Field(default=MemoryType.SHORT)
    db_path: str = Field(default="weakagent.sqlite3")
    max_messages: int = Field(default=100)

    run_id: str = Field(default_factory=lambda: f"srun_{uuid.uuid4().hex[:12]}")
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_type: Optional[str] = None
    agent_id: Optional[str] = None
    status: str = Field(default="completed")

    # Cleanup configuration
    cleanup_strategy: MemoryCleanupStrategy = Field(
        default=MemoryCleanupStrategy.KEEP_LAST_N
    )
    keep_last_n: int = Field(default=12)
    truncate_tool_chars: int = Field(default=2000)
    summarize_keep_last_n: int = Field(default=20)
    enable_token_window_cleanup: bool = Field(default=True)
    enable_message_limit_cleanup: bool = Field(default=True)
    max_context_turns: int = Field(
        default=30,
        description="Maximum complete conversation turns before trimming.",
    )
    compress_turn_threshold: int = Field(default=5)
    max_history_tool_chars: int = Field(default=20000)

    def _init_db(self) -> None:
        self.ensure_db_parent()
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS short_memory_snapshot (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_id TEXT UNIQUE NOT NULL,
                    run_id TEXT NOT NULL,
                    session_id TEXT,
                    user_id TEXT,
                    agent_type TEXT,
                    agent_id TEXT,
                    status TEXT DEFAULT 'completed',
                    messages TEXT NOT NULL,
                    messages_count INTEGER DEFAULT 0,
                    extra TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(run_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_short_memory_snapshot_run_id "
                "ON short_memory_snapshot(run_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_short_memory_snapshot_session_id "
                "ON short_memory_snapshot(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_short_memory_snapshot_created_at "
                "ON short_memory_snapshot(created_at)"
            )
            conn.commit()

    def save_snapshot(
        self,
        *,
        run_id: Optional[str] = None,
        messages: Optional[List[Message]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist the full short-memory message list for this run."""
        msgs = messages if messages is not None else self.messages
        if not msgs:
            logger.warning("Skip short memory snapshot save: no messages")
            return ""

        rid = run_id or self.run_id
        snapshot_id = f"snap_{uuid.uuid4().hex[:16]}"
        payload = dict(extra or {})
        payload.setdefault("cleanup_strategy", self.cleanup_strategy.value)

        messages_json = json.dumps(
            [m.to_dict() for m in msgs],
            ensure_ascii=False,
        )

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO short_memory_snapshot(
                    snapshot_id, run_id, session_id, user_id, agent_type,
                    agent_id, status, messages, messages_count, extra, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    snapshot_id = excluded.snapshot_id,
                    session_id = excluded.session_id,
                    user_id = excluded.user_id,
                    agent_type = excluded.agent_type,
                    agent_id = excluded.agent_id,
                    status = excluded.status,
                    messages = excluded.messages,
                    messages_count = excluded.messages_count,
                    extra = excluded.extra,
                    created_at = excluded.created_at
                """,
                (
                    snapshot_id,
                    rid,
                    self.session_id,
                    self.user_id,
                    self.agent_type,
                    self.agent_id,
                    self.status,
                    messages_json,
                    len(msgs),
                    json.dumps(payload, ensure_ascii=False),
                    self.utc_now_iso(),
                ),
            )
            conn.commit()

        logger.info(
            "Short memory snapshot saved run_id=%s messages=%s",
            rid,
            len(msgs),
        )
        return rid

    @classmethod
    def _messages_from_json(cls, raw: str) -> List[Message]:
        data = json.loads(raw or "[]")
        if not isinstance(data, list):
            return []
        out: List[Message] = []
        for item in data:
            if isinstance(item, dict):
                out.append(Message.model_validate(item))
        return out

    @classmethod
    def _row_to_entry(cls, row: Any) -> tuple[ShortMemorySnapshotEntry, List[Message]]:
        try:
            extra = json.loads(row["extra"] or "{}")
        except Exception:
            extra = {}
        if not isinstance(extra, dict):
            extra = {}
        entry = ShortMemorySnapshotEntry(
            snapshot_id=str(row["snapshot_id"]),
            run_id=str(row["run_id"]),
            session_id=row["session_id"],
            user_id=row["user_id"],
            agent_type=row["agent_type"],
            agent_id=row["agent_id"],
            status=str(row["status"] or "completed"),
            messages_count=int(row["messages_count"] or 0),
            extra=extra,
            created_at=row["created_at"],
        )
        messages = cls._messages_from_json(str(row["messages"] or "[]"))
        return entry, messages

    @classmethod
    def fetch_snapshot(
        cls,
        run_id: str,
        *,
        db_path: Optional[str] = None,
    ) -> Optional[tuple[ShortMemorySnapshotEntry, List[Message]]]:
        path = str(cls._resolve_db_path(db_path or "weakagent.sqlite3", MemoryType.SHORT))
        with cls._connect_db(path) as conn:
            row = conn.execute(
                """
                SELECT snapshot_id, run_id, session_id, user_id, agent_type,
                       agent_id, status, messages, messages_count, extra, created_at
                FROM short_memory_snapshot
                WHERE run_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
        return cls._row_to_entry(row) if row else None

    def load_snapshot(
        self,
        run_id: Optional[str] = None,
        *,
        clear: bool = True,
    ) -> List[Message]:
        """Load a persisted snapshot into ``messages``."""
        rid = run_id or self.run_id
        result = self.fetch_snapshot(rid, db_path=self.db_path)
        if not result:
            return []
        _entry, messages = result
        if clear:
            self.messages.clear()
        self.add_messages(messages)
        return list(self.messages)

    @classmethod
    def list_snapshots(
        cls,
        *,
        db_path: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[ShortMemorySnapshotEntry]:
        path = str(cls._resolve_db_path(db_path or "weakagent.sqlite3", MemoryType.SHORT))
        query = """
            SELECT snapshot_id, run_id, session_id, user_id, agent_type,
                   agent_id, status, messages_count, extra, created_at
            FROM short_memory_snapshot
            WHERE 1=1
        """
        params: List[Any] = []
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(max(1, int(limit)))

        with cls._connect_db(path) as conn:
            rows = conn.execute(query, params).fetchall()
        entries: List[ShortMemorySnapshotEntry] = []
        for row in rows:
            try:
                extra = json.loads(row["extra"] or "{}")
            except Exception:
                extra = {}
            if not isinstance(extra, dict):
                extra = {}
            entries.append(
                ShortMemorySnapshotEntry(
                    snapshot_id=str(row["snapshot_id"]),
                    run_id=str(row["run_id"]),
                    session_id=row["session_id"],
                    user_id=row["user_id"],
                    agent_type=row["agent_type"],
                    agent_id=row["agent_id"],
                    status=str(row["status"] or "completed"),
                    messages_count=int(row["messages_count"] or 0),
                    extra=extra,
                    created_at=row["created_at"],
                )
            )
        return entries

    @classmethod
    def _connect_db(cls, db_path: str):
        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ---- Cleanup helpers ----
    def _keep_system_prefix(self, msgs: List[Message]) -> tuple[list[Message], list[Message]]:
        sys_prefix: list[Message] = []
        rest: list[Message] = []
        for m in msgs:
            if not rest and m.role == "system":
                sys_prefix.append(m)
            else:
                rest.append(m)
        return sys_prefix, rest

    def _identify_complete_turns(self) -> List[dict]:
        _, rest = self._keep_system_prefix(self.messages)
        if not rest:
            return []

        user_indices = [i for i, m in enumerate(rest) if m.role == "user"]
        if not user_indices:
            return [{"messages": rest}]

        turns: List[dict] = []
        for idx, start in enumerate(user_indices):
            end = user_indices[idx + 1] if idx + 1 < len(user_indices) else len(rest)
            turn_msgs = rest[start:end]
            if turn_msgs:
                turns.append({"messages": turn_msgs})
        return turns

    def _rebuild_from_turns(self, turns: List[dict]) -> None:
        sys_prefix, _ = self._keep_system_prefix(self.messages)
        new_messages: list[Message] = []
        for t in turns:
            new_messages.extend(t.get("messages", []))
        self.messages = sys_prefix + new_messages

    def _compress_turn_to_text_only(self, turn: dict) -> dict:
        msgs: List[Message] = list(turn.get("messages", []))
        if not msgs:
            return {"messages": []}

        user_msg = next((m for m in msgs if m.role == "user"), None)
        assistant_msgs = [m for m in msgs if m.role == "assistant"]
        last_assistant = None
        for m in reversed(assistant_msgs):
            if (m.content or "").strip():
                last_assistant = m
                break
        if last_assistant is None and assistant_msgs:
            last_assistant = assistant_msgs[-1]

        kept: List[Message] = []
        if user_msg is not None:
            kept.append(user_msg)
        if last_assistant is not None:
            kept.append(last_assistant.model_copy(update={"tool_calls": None}))  # type: ignore
        return {"messages": kept}

    def _trim_keep_last_n(self, n: int) -> bool:
        n = max(1, int(n))
        turns = self._identify_complete_turns()
        if len(turns) <= n:
            return False
        before = len(self.messages)
        self._rebuild_from_turns(turns[-n:])
        logger.warning(
            "Memory cleanup: keep_last_n trimmed turns. before=%s after=%s keep_last_n=%s",
            before,
            len(self.messages),
            n,
        )
        return True

    def _truncate_tool_outputs(self, max_chars: int) -> bool:
        max_chars = max(200, int(max_chars))
        changed = False
        new_msgs: list[Message] = []
        for m in self.messages:
            if m.role == "tool" and isinstance(m.content, str) and len(m.content) > max_chars:
                new_msgs.append(m.with_truncated_content_middle(max_chars=max_chars))
                changed = True
            else:
                new_msgs.append(m)
        if changed:
            self.messages = new_msgs
            logger.warning("Memory cleanup: truncated tool outputs. max_chars=%s", max_chars)
        return changed

    def _truncate_historical_tool_outputs(self) -> bool:
        limit = max(500, int(self.max_history_tool_chars))
        if len(self.messages) < 2:
            return False

        sys_prefix, rest = self._keep_system_prefix(self.messages)
        if not rest:
            return False
        last_user_idx = None
        for i in range(len(rest) - 1, -1, -1):
            if rest[i].role == "user":
                last_user_idx = i
                break
        current_turn_start = last_user_idx if last_user_idx is not None else len(rest)

        truncated = 0
        new_rest: list[Message] = []
        for i, m in enumerate(rest):
            if i >= current_turn_start:
                new_rest.append(m)
                continue
            if m.role == "tool" and isinstance(m.content, str) and len(m.content) > limit:
                new_rest.append(m.with_truncated_content_middle(max_chars=limit))
                truncated += 1
            else:
                new_rest.append(m)

        if truncated:
            self.messages = sys_prefix + new_rest
            logger.warning(
                "Memory cleanup: truncated %s historical tool output(s) to %s chars",
                truncated,
                limit,
            )
            return True
        return False

    async def _summarize_then_keep_last_n(self, llm, keep_last_n: int) -> bool:
        keep_last_n = max(2, int(keep_last_n))
        sys_prefix, rest = self._keep_system_prefix(self.messages)
        if len(rest) <= keep_last_n:
            return False

        old = rest[:-keep_last_n]
        recent = rest[-keep_last_n:]

        summary_msg = await summarize_short_memory(llm, old)
        summary_msg = Message.assistant_message(
            f"[Summary of earlier conversation]\n{summary_msg.content or ''}"
        )

        before = len(self.messages)
        self.messages = sys_prefix + [summary_msg] + recent
        logger.warning(
            "Memory cleanup: summarized old messages. before=%s after=%s summarized=%s kept_recent=%s",
            before,
            len(self.messages),
            len(old),
            keep_last_n,
        )
        return True

    def _exceeds_token_window(self, llm) -> bool:
        if not self.enable_token_window_cleanup:
            return False
        context_window = getattr(llm, "context_window", None)
        if not context_window:
            return False
        reserve = getattr(llm, "reserve_completion_tokens", None)
        if reserve is None:
            reserve = getattr(llm, "max_tokens", 0)
        budget = max(1, int(context_window) - int(reserve))

        formatted = llm.format_messages(
            self.messages, supports_images=getattr(llm, "supports_images", False)
        )
        total = llm.count_message_tokens(formatted)
        if total > budget:
            logger.warning(
                "Memory token window exceeded. total=%s budget=%s context_window=%s reserve=%s",
                total,
                budget,
                context_window,
                reserve,
            )
            return True
        return False

    def _drop_older_half_turns(self) -> bool:
        turns = self._identify_complete_turns()
        if len(turns) < 2:
            return False
        removed = len(turns) // 2
        kept = turns[-(len(turns) - removed) :]
        before = len(self.messages)
        self._rebuild_from_turns(kept)
        logger.warning(
            "Memory cleanup: dropped older half turns. before_turns=%s after_turns=%s removed=%s (messages %s -> %s)",
            len(turns),
            len(kept),
            removed,
            before,
            len(self.messages),
        )
        return True

    def _compress_all_turns_to_text_only(self) -> bool:
        turns = self._identify_complete_turns()
        if not turns:
            return False
        before = len(self.messages)
        compressed = [self._compress_turn_to_text_only(t) for t in turns]
        self._rebuild_from_turns(compressed)
        logger.warning(
            "Memory cleanup: compressed all turns to text-only. turns=%s messages %s -> %s",
            len(turns),
            before,
            len(self.messages),
        )
        return True

    async def cleanup_if_needed(self, *, llm) -> bool:
        """Cleanup memory before a step/LLM call. Returns True if memory changed."""
        changed = False
        changed |= self._truncate_historical_tool_outputs()

        if self.cleanup_strategy == MemoryCleanupStrategy.TRUNCATE_TOOL_OUTPUT:
            changed |= self._truncate_tool_outputs(self.truncate_tool_chars)
            turns = self._identify_complete_turns()
            if len(turns) > self.max_context_turns:
                changed |= self._drop_older_half_turns()
            if self._exceeds_token_window(llm):
                turns = self._identify_complete_turns()
                if len(turns) < self.compress_turn_threshold:
                    changed |= self._compress_all_turns_to_text_only()
                else:
                    changed |= self._drop_older_half_turns()
            return changed

        if self.cleanup_strategy == MemoryCleanupStrategy.SUMMARIZE_THEN_KEEP_LAST_N:
            turns = self._identify_complete_turns()
            if len(turns) > self.max_context_turns:
                return await self._summarize_then_keep_last_n(llm, self.summarize_keep_last_n)
            if self._exceeds_token_window(llm):
                return await self._summarize_then_keep_last_n(llm, self.summarize_keep_last_n)
            return changed

        turns = self._identify_complete_turns()
        if len(turns) > self.max_context_turns:
            changed |= self._drop_older_half_turns()
            return changed

        if self._exceeds_token_window(llm):
            turns = self._identify_complete_turns()
            if len(turns) < self.compress_turn_threshold:
                changed |= self._compress_all_turns_to_text_only()
            else:
                changed |= self._drop_older_half_turns()
            return changed

        if self.enable_message_limit_cleanup and len(self.messages) > self.max_messages:
            changed |= self._trim_keep_last_n(self.keep_last_n)
        return changed
