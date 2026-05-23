from __future__ import annotations

import json
from typing import List, Optional

from pydantic import Field, model_validator

from weakagent.llm.llm import LLM
from weakagent.memory.base import MemoryType
from weakagent.memory.conversation import select_last_n_messages_with_integrity
from weakagent.memory.runtime_session import RuntimeSessionStore
from weakagent.schemas.message import Message
from weakagent.utils.logger import get_logger

logger = get_logger(__name__)


class RuntimeMemory(RuntimeSessionStore):
    """Agent-level runtime memory backed by `runtime_session*` sqlite tables.

    Keeps request + last_result pairs in memory and persists each append.
    """

    memory_type: MemoryType = Field(default=MemoryType.RUNTIME)
    loaded_session_id: Optional[str] = Field(
        default=None,
        description="Session id from the last successful load_from_session call",
    )

    @model_validator(mode="after")
    def _init_runtime_memory(self) -> "RuntimeMemory":
        # RuntimeSessionStore._init already ran via MRO; keep loaded_session_id in sync.
        self.loaded_session_id = self.session_id
        return self

    def add_request(self, request: Optional[str], *, extra: Optional[dict] = None) -> None:
        if not request:
            return
        msg = Message.user_message(request)
        self.add_message(msg)
        try:
            self.ensure_session()
            self._persist_message(msg, extra=extra)
        except Exception:
            logger.exception("Failed to persist runtime session request message")

    def format_for_long_memory_extraction(self) -> str:
        """Serialize in-memory runtime messages for long-memory LLM extraction."""
        payload = [
            {"role": str(m.role), "content": m.content or ""}
            for m in self.messages
        ]
        return json.dumps(payload, ensure_ascii=False)

    def add_last_result(self, last_result: Optional[str], *, extra: Optional[dict] = None) -> None:
        if last_result is None:
            return
        msg = Message.assistant_message(last_result)
        self.add_message(msg)
        try:
            self.ensure_session()
            self._persist_message(msg, extra=extra)
        except Exception:
            logger.exception("Failed to persist runtime session result message")

    def load_from_session(
        self,
        session_id: Optional[str] = None,
        *,
        last_n: Optional[int] = None,
        clear: bool = True,
        switch_session: bool = True,
    ) -> List[Message]:
        """Load raw runtime session messages from sqlite into memory.

        Args:
            last_n: Keep the last N non-system messages (tool chains preserved).
                ``None`` loads all messages. Ignored when negative.
            switch_session: If True, set ``session_id`` to the loaded session.
                Use False when resuming context into a new session row.
        """
        sid = session_id or self.get_last_session_id(self.db_path)
        if not sid:
            logger.warning("No runtime session found in db_path=%s", self.db_path)
            if clear:
                self.messages.clear()
            self.loaded_session_id = None
            return []

        try:
            loaded = self.fetch_session_messages(sid, db_path=self.db_path)
        except Exception:
            logger.exception(
                "Failed to load runtime session messages session_id=%s", sid
            )
            raise

        if last_n is not None and last_n > 0:
            loaded = select_last_n_messages_with_integrity(loaded, last_n)

        if clear:
            self.messages.clear()
        self.add_messages(loaded)
        if switch_session:
            self.session_id = sid
        self.loaded_session_id = sid
        return list(self.messages)

    def load_last_runtime_session(
        self,
        *,
        last_n: int = 10,
        clear: bool = True,
    ) -> List[Message]:
        """Load raw messages from the most recent prior runtime session (not summaries).

        Args:
            last_n: Number of recent messages to load (default 10). Use ``-1`` for all.
        """
        common = dict(
            db_path=self.db_path,
            exclude_session_id=self.session_id,
            require_messages=True,
        )
        sid = self.get_last_session_id(agent_id=self.agent_id, **common)
        matched_by = "agent_id" if sid else None
        if not sid and self.agent_type:
            sid = self.get_last_session_id(agent_type=self.agent_type, **common)
            matched_by = "agent_type" if sid else matched_by
        if not sid and self.user_id:
            sid = self.get_last_session_id(user_id=self.user_id, **common)
            matched_by = "user_id" if sid else matched_by
        if not sid:
            sid = self.get_last_session_id(**common)
            matched_by = "global" if sid else None
        if not sid:
            logger.info(
                "No previous runtime session to load (agent_id=%s agent_type=%s user_id=%s)",
                self.agent_id,
                self.agent_type,
                self.user_id,
            )
            if clear:
                self.messages.clear()
            return []
        logger.info(
            "Resolved previous runtime session_id=%s via %s",
            sid,
            matched_by,
        )

        effective_last_n = None if last_n < 0 else last_n
        loaded = self.load_from_session(
            sid,
            last_n=effective_last_n,
            clear=clear,
            switch_session=False,
        )
        logger.info(
            "Loaded %s raw runtime session message(s) from session_id=%s (last_n=%s)",
            len(loaded),
            sid,
            last_n,
        )
        return loaded

    def load_last_n_messages(self, n: int, *, clear: bool = True) -> List[Message]:
        if n == 0:
            return []
        sid = self.loaded_session_id or self.get_last_session_id(self.db_path)
        effective_last_n = None if n < 0 else n
        return self.load_from_session(sid, last_n=effective_last_n, clear=clear)

    async def finalize_session(
        self,
        *,
        status: str = "closed",
        run_id: str = "loop_end",
        llm: Optional[LLM] = None,
        extra: Optional[dict] = None,
    ) -> str:
        """Mark session closed, ensure title, and write loop-level summary."""
        self.status = status
        self.ensure_session()

        if not self.title:
            first_user = next((m for m in self.messages if m.role == "user" and m.content), None)
            if first_user:
                await self.generate_title_from_request(first_user.content or "", llm=llm)

        llm = llm or LLM(config_name="fast")
        payload = {"source": "runtime_loop_finalize"}
        if extra:
            payload.update(extra)
        summary = await self.write_session_summary(
            run_id=run_id,
            status=status,
            messages=list(self.messages),
            llm=llm,
            extra=payload,
        )
        self.ensure_session()
        return summary
