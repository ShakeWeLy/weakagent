from abc import ABC
from enum import Enum
from typing import List
from pydantic import BaseModel, Field
from weakagent.schemas.message import Message
from weakagent.schemas.tool import ToolCall
from weakagent.utils.logger import get_logger

logger = get_logger(__name__)

class MemoryType(str, Enum):
    SHORT = "short"
    LONG = "long"
    WORKING = "working"


class MemoryCleanupStrategy(str, Enum):
    KEEP_LAST_N = "keep_last_n"
    TRUNCATE_TOOL_OUTPUT = "truncate_tool_output"
    SUMMARIZE_THEN_KEEP_LAST_N = "summarize_then_keep_last_n"



class BaseMemory(BaseModel, ABC):
    messages: List[Message] = Field(default_factory=list)
    max_messages: int = Field(default=100)
    memory_type: MemoryType = Field(default=MemoryType.SHORT)

    # Cleanup configuration
    cleanup_strategy: MemoryCleanupStrategy = Field(
        default=MemoryCleanupStrategy.KEEP_LAST_N
    )
    # When trimming by turns, keep last N turns (not raw messages).
    keep_last_n: int = Field(default=12)
    truncate_tool_chars: int = Field(default=2000)
    summarize_keep_last_n: int = Field(default=20)
    enable_token_window_cleanup: bool = Field(default=True)
    enable_message_limit_cleanup: bool = Field(default=True)
    max_context_turns: int = Field(
        default=30,
        description="Maximum number of complete conversation turns to keep in memory.",
    )
    # For token window trimming: if turns are fewer than this threshold, compress turns to text-only rather than discarding.
    compress_turn_threshold: int = Field(default=5)
    # Historical tool outputs can be further truncated before token trimming.
    max_history_tool_chars: int = Field(default=20000)

    def add_message(self, message: Message) -> None:
        """Add a message to memory"""
        self.messages.append(message)
        # Optional: Implement message limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def add_messages(self, messages: List[Message]) -> None:
        """Add multiple messages to memory"""
        self.messages.extend(messages)
        # Optional: Implement message limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]
    
    def add_messages_from_tool_calls(self, tool_calls: List[ToolCall]) -> None:
        """Add messages from tool calls"""
        for tool_call in tool_calls:
            self.add_message(Message.tool_message(
                content=tool_call.function.arguments,
                name=tool_call.function.name,
                tool_call_id=tool_call.id
            ))

    def clear(self) -> None:
        """Clear all messages"""
        self.messages.clear()
    
    def clear_messages(self, n: int) -> None:
        """Clear n messages"""
        self.messages = self.messages[:-n]
        logger.info(f"clear meassage done, now: {self.messages}")


    def get_recent_messages(self, n: int) -> List[Message]:
        """Get n most recent messages"""
        return self.messages[-n:]

    def to_dict_list(self) -> List[dict]:
        """Convert messages to list of dicts"""
        return [msg.to_dict() for msg in self.messages]

    # ---- Cleanup helpers ----
    def _keep_system_prefix(self, msgs: List[Message]) -> tuple[list[Message], list[Message]]:
        """Split leading system messages (preserve order)."""
        sys_prefix: list[Message] = []
        rest: list[Message] = []
        for m in msgs:
            if not rest and m.role == "system":
                sys_prefix.append(m)
            else:
                rest.append(m)
        return sys_prefix, rest

    def _identify_complete_turns(self) -> List[dict]:
        """
        Identify complete conversation turns as a unit for trimming.

        A "turn" is defined as:
        - starts at a user message
        - includes everything until (but not including) the next user message
        This preserves tool chains (assistant tool_calls + tool results) without mid-chain truncation.
        """
        _, rest = self._keep_system_prefix(self.messages)
        if not rest:
            return []

        # Find indices of user messages.
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

    def _estimate_turn_tokens(self, llm, turn: dict) -> int:
        msgs = turn.get("messages", [])
        formatted = llm.format_messages(
            msgs, supports_images=getattr(llm, "supports_images", False)
        )
        return llm.count_message_tokens(formatted)

    def _compress_turn_to_text_only(self, turn: dict) -> dict:
        """
        Compress a turn by stripping tool messages and tool_calls payloads.
        Keeps:
        - the leading user message
        - the last assistant message (prefer one with non-empty content)
        """
        msgs: List[Message] = list(turn.get("messages", []))
        if not msgs:
            return {"messages": []}

        user_msg = None
        for m in msgs:
            if m.role == "user":
                user_msg = m
                break

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
            # Drop tool_calls payload to save tokens; keep content only.
            kept.append(last_assistant.model_copy(update={"tool_calls": None}))  # type: ignore

        return {"messages": kept}

    def _trim_keep_last_n(self, n: int) -> bool:
        """Trim memory to keep last N complete turns (system prefix preserved)."""
        n = max(1, int(n))
        turns = self._identify_complete_turns()
        if len(turns) <= n:
            return False
        before = len(self.messages)
        kept = turns[-n:]
        self._rebuild_from_turns(kept)
        logger.warning(
            "Memory cleanup: keep_last_n trimmed turns. before=%s after=%s keep_last_n=%s",
            before,
            len(self.messages),
            n,
        )
        return True

    def _truncate_tool_outputs(self, max_chars: int) -> bool:
        """Truncate long tool outputs in-place."""
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
        """
        Further truncate tool outputs in historical turns (excluding the current turn)
        to reduce context size before token-based trimming.
        """
        limit = max(500, int(self.max_history_tool_chars))
        if len(self.messages) < 2:
            return False

        # Current turn boundary: last user message in non-system rest.
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
        """
        Summarize older messages into a single assistant summary, then keep last N.
        """
        from weakagent.llm.summarize import summarize_working_memory

        keep_last_n = max(2, int(keep_last_n))
        sys_prefix, rest = self._keep_system_prefix(self.messages)
        if len(rest) <= keep_last_n:
            return False

        old = rest[:-keep_last_n]
        recent = rest[-keep_last_n:]

        summary_msg = await summarize_working_memory(llm, old)
        summary_msg = Message.assistant_message(
            f"[Summary of earlier conversation]\n{summary_msg.content}"
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
        """
        Check whether current memory messages exceed the LLM token budget for prompt.
        """
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

    async def cleanup_if_needed(self, *, llm) -> bool:
        """
        Cleanup memory before a step/LLM call.
        Returns True if memory changed.
        """
        changed = False

        # Step 0: shrink oversized historical tool outputs first (may avoid dropping turns)
        changed |= self._truncate_historical_tool_outputs()

        if self.cleanup_strategy == MemoryCleanupStrategy.TRUNCATE_TOOL_OUTPUT:
            changed |= self._truncate_tool_outputs(self.truncate_tool_chars)
            # Turn limit
            turns = self._identify_complete_turns()
            if len(turns) > self.max_context_turns:
                removed = len(turns) // 2
                kept = turns[-(len(turns) - removed) :]
                before = len(self.messages)
                self._rebuild_from_turns(kept)
                logger.warning(
                    "Memory cleanup: turns exceeded max_context_turns. before_turns=%s after_turns=%s removed=%s (messages %s -> %s)",
                    len(turns),
                    len(kept),
                    removed,
                    before,
                    len(self.messages),
                )
                changed = True

            # Token window
            if self._exceeds_token_window(llm):
                turns = self._identify_complete_turns()
                if len(turns) < self.compress_turn_threshold:
                    compressed = [self._compress_turn_to_text_only(t) for t in turns]
                    before = len(self.messages)
                    self._rebuild_from_turns(compressed)
                    logger.warning(
                        "Memory cleanup: token overflow with few turns, compressed all turns to text-only. turns=%s messages %s -> %s",
                        len(turns),
                        before,
                        len(self.messages),
                    )
                    changed = True
                else:
                    removed = len(turns) // 2
                    kept = turns[-(len(turns) - removed) :]
                    before = len(self.messages)
                    self._rebuild_from_turns(kept)
                    logger.warning(
                        "Memory cleanup: token overflow, dropped older half turns. before_turns=%s after_turns=%s removed=%s (messages %s -> %s)",
                        len(turns),
                        len(kept),
                        removed,
                        before,
                        len(self.messages),
                    )
                    changed = True
            return changed

        if self.cleanup_strategy == MemoryCleanupStrategy.SUMMARIZE_THEN_KEEP_LAST_N:
            turns = self._identify_complete_turns()
            if len(turns) > self.max_context_turns:
                changed |= await self._summarize_then_keep_last_n(llm, self.summarize_keep_last_n)
                return changed
            if self._exceeds_token_window(llm):
                changed |= await self._summarize_then_keep_last_n(llm, self.summarize_keep_last_n)
                return changed
            return changed

        # Default: KEEP_LAST_N
        turns = self._identify_complete_turns()
        if len(turns) > self.max_context_turns:
            removed = len(turns) // 2
            kept = turns[-(len(turns) - removed) :]
            before = len(self.messages)
            self._rebuild_from_turns(kept)
            logger.warning(
                "Memory cleanup: turns exceeded max_context_turns. before_turns=%s after_turns=%s removed=%s (messages %s -> %s)",
                len(turns),
                len(kept),
                removed,
                before,
                len(self.messages),
            )
            changed = True
            return changed

        if self._exceeds_token_window(llm):
            turns = self._identify_complete_turns()
            if len(turns) < self.compress_turn_threshold:
                compressed = [self._compress_turn_to_text_only(t) for t in turns]
                before = len(self.messages)
                self._rebuild_from_turns(compressed)
                logger.warning(
                    "Memory cleanup: token overflow with few turns, compressed all turns to text-only. turns=%s messages %s -> %s",
                    len(turns),
                    before,
                    len(self.messages),
                )
                changed = True
                return changed

            removed = len(turns) // 2
            kept = turns[-(len(turns) - removed) :]
            before = len(self.messages)
            self._rebuild_from_turns(kept)
            logger.warning(
                "Memory cleanup: token overflow, dropped older half turns. before_turns=%s after_turns=%s removed=%s (messages %s -> %s)",
                len(turns),
                len(kept),
                removed,
                before,
                len(self.messages),
            )
            changed = True
            return changed

        # Fallback message count limit (legacy hard cap)
        if self.enable_message_limit_cleanup and len(self.messages) > self.max_messages:
            changed |= self._trim_keep_last_n(self.keep_last_n)
            return changed
        return changed