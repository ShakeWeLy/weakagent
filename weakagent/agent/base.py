from abc import ABC, abstractmethod
import asyncio
from contextlib import asynccontextmanager
import inspect
import threading
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from weakagent.llm import LLM
from weakagent.utils.logger import get_logger
from weakagent.schemas.message import ROLE_TYPE, Message, Role
from weakagent.memory.conversation import ConversationMemory
from weakagent.memory.short import ShortMemory
from weakagent.memory.long import LongMemory
from weakagent.memory.working import WorkingMemory
from weakagent.memory.session import SessionMemory, SessionMemorySummaryEntry
from weakagent.schemas.agent import AgentState

logger = get_logger(__name__)

EventCallback = Callable[[dict], Union[Any, Awaitable[Any]]]


class BaseAgent(BaseModel, ABC):
    """Abstract base class for managing agent state and execution.

    Provides foundational functionality for state transitions, memory management,
    and a step-based execution loop. Subclasses must implement the `step` method.
    """
    # Core attributes
    name: str = Field(..., description="Unique name of the agent")
    description: Optional[str] = Field(None, description="Optional agent description")

    # Prompts
    system_prompt: Optional[str] = Field(
        None, description="System-level instruction prompt"
    )
    next_step_prompt: Optional[str] = Field(
        None, description="Prompt for determining next action"
    )

    # Dependencies
    llm: LLM = Field(default_factory=LLM, description="Language model instance")
    working_memory: WorkingMemory = Field(
        default_factory=WorkingMemory, description="Per-run agent working context"
    )
    short_memory: ShortMemory = Field(
        default_factory=ShortMemory, description="Per-run agent working context with history"
    )
    conversation: Optional[ConversationMemory] = Field(
        default=None,
        description="Append-only per-message persistence",
    )
    session: Optional[SessionMemory] = Field(
        default=None,
        description="Runtime-scoped session metadata and end-of-loop summary",
    )
    long_memory: Optional[LongMemory] = Field(
        default=None,
        description="Long-term memory message for prompt injection",
    )
    long_memory_message: Optional[Message] = Field(
        default=None,
        description="Long-term memory message for prompt injection",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="User id for the agent",
    )
    awaiting_human: bool = Field(
        default=False,
        description="If true, this run is paused awaiting user input; do not persist last_result.",
    )
    # Single-run-level 
    run_id: Optional[str] = Field(
        default_factory=lambda: uuid.uuid4().hex[:12],
        description="Run id for the agent",
    )
    state: AgentState = Field(
        default=AgentState.IDLE, description="Current agent state"
    )

    # session-level persistence helpers (last_request/last_result for each agent.run).
    session_id: Optional[str] = Field(
        default_factory=lambda: uuid.uuid4().hex[:12],
        description="Session id for the agent",
    )
    agent_id: Optional[str] = Field(
        default_factory=lambda: uuid.uuid4().hex[:12],
        description="Agent id for the agent",
    )
    last_request: Optional[str] = Field(default=None)
    last_result: Optional[str] = Field(default=None)

    on_event: Optional[EventCallback] = Field(
        default=None, description="Optional event callback for agent lifecycle events"
    )

    # Execution control
    max_steps: int = Field(default=10, description="Maximum steps before termination")
    current_step: int = Field(default=0, description="Current step in execution")
    duplicate_threshold: int = 2

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )
    
    # Optional
    only_last_result: bool = Field(default="", description="If only return last output result from agent")
    summarize_short_memory: bool = Field(default=False, description="If return the summarize of the short memory")
    summarize_working_memory: bool = Field(default=False, description="If summarize the working memory to enable SKILLS_USAGE_PROMPT")
    verbose: bool = Field(default=False, description="If verbose the agent execution, llm input and output")
    use_long_memory: bool = Field(default=False, description="If use long-term memory")
    only_save_last_result_to_short: bool = Field(default=False, description="If only save the last result to the short memory")
    skills_enabled: bool = Field(
        default=True, description="Inject <available_skills> into system prompts"
    )
    skill_filter: Optional[List[str]] = Field(
        default=None, description="Optional allow-list of skill names"
    )

    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        """Initialize agent with default settings if not provided."""
        try:
            if self.llm is None or not isinstance(self.llm, LLM):
                self.llm = LLM(config_name=self.name.lower())
            if not isinstance(self.short_memory, ShortMemory):
                self.short_memory = ShortMemory(
                    run_id=self.run_id,
                    user_id=self.user_id,
                    agent_type=self.name,
                    agent_id=self.agent_id,
                    session_id=self.session_id,

                )
            if self.session is None or not isinstance(self.session, SessionMemory):
                self.session = SessionMemory(
                    run_id=self.run_id,
                    user_id=self.user_id,
                    agent_type=self.name,
                    agent_id=self.agent_id,
                    session_id=self.session_id,

                )
            if self.conversation is None:
                self.conversation = ConversationMemory(
                    run_id=self.run_id,
                    user_id=self.user_id,
                    agent_type=self.name,
                    agent_id=self.agent_id,
                    session_id=self.session_id,
                    )

            if self.use_long_memory:
                if self.long_memory is None:
                    self.long_memory = LongMemory(user_id=self.user_id)
                else:
                    self.long_memory.user_id = self.user_id
                self.long_memory.load_for_user(self.user_id)
                self.update_memory("system", self.long_memory.to_system_message().content)
            return self
        except Exception:
            logger.exception("Failed to initialize agent")
            return self

    def get_skill_manager(self):
        """Lazy-init SkillManager (not a pydantic field to avoid deep copy issues)."""
        mgr = getattr(self, "_skill_manager", None)
        if mgr is None:
            from weakagent.skills.manager import SkillManager

            mgr = SkillManager()
            object.__setattr__(self, "_skill_manager", mgr)
        return mgr

    def with_skills_prompt(self, base_prompt: Optional[str]) -> Optional[str]:
        """Append skills usage instructions and `<available_skills>` to a system prompt."""
        if not base_prompt or not self.skills_enabled:
            return base_prompt

        mgr = self.get_skill_manager()
        block = mgr.build_skills_prompt(skill_filter=self.skill_filter)
        if not block.strip():
            return base_prompt

        from weakagent.skills.prompt import SKILLS_USAGE_PROMPT

        return f"{base_prompt.rstrip()}\n\n{SKILLS_USAGE_PROMPT.strip()}\n{block}"

    def _emit_event(self, event_type: str, data: Optional[dict] = None) -> None:
        """Emit an event to callbacks (sync/async; isolated failures).

        Delivery order:
        1) self.on_event (if set)
        2) runtime.on_event (if agent is runtime-managed and runtime exposes on_event)
        """

        event = {
            "type": event_type,
            "timestamp": time.time(),
            "data": data or {},
        }

        def _dispatch(cb: Any) -> None:
            if cb is None:
                return
            try:
                result = cb(event)
                if inspect.isawaitable(result):
                    task = asyncio.create_task(result)

                    def _on_done(t: asyncio.Task) -> None:
                        try:
                            exc = t.exception()
                        except asyncio.CancelledError:
                            return
                        except Exception:
                            logger.exception(
                                "Unexpected error while checking on_event task"
                            )
                            return
                        if exc is not None:
                            logger.exception("on_event async callback failed", exc_info=exc)

                    task.add_done_callback(_on_done)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

        _dispatch(self.on_event)

        runtime = getattr(self, "agent_runtime", None)
        runtime_cb = getattr(runtime, "on_event", None) if runtime is not None else None
        if runtime_cb is not self.on_event:
            _dispatch(runtime_cb)

    def append_message(self, message: Message, *, extra: Optional[Dict[str, Any]] = None) -> None:
        """Append to short memory, conversation log, and in-memory session transcript."""
        self.working_memory.add_message(message)
        self.short_memory.add_message(message)
        persist_extra = {
            "agent": self.name,
            "state": str(self.state),
            "current_step": self.current_step,
        }
        if extra:
            persist_extra.update(extra)
        if self.conversation is not None:
            if self.run_id:
                self.conversation.run_id = self.run_id
            self.conversation.add_message(message, extra=persist_extra)

    @asynccontextmanager
    async def state_context(self, new_state: AgentState):
        """Context manager for safe agent state transitions.

        Args:
            new_state: The state to transition to during the context.

        Yields:
            None: Allows execution within the new state.

        Raises:
            ValueError: If the new_state is invalid.
        """
        if not isinstance(new_state, AgentState):
            raise ValueError(f"Invalid state: {new_state}")

        previous_state = self.state
        self.state = new_state
        try:
            yield
        except Exception as e:
            self.state = AgentState.ERROR  # Transition to ERROR on failure
            raise e
        finally:
            self.state = previous_state  # Revert to previous state

    def update_memory(
        self,
        role: ROLE_TYPE,  # type: ignore
        content: str,
        base64_image: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Add a message to the agent's memory.

        Args:
            role: The role of the message sender (user, system, assistant, tool).
            content: The message content.
            base64_image: Optional base64 encoded image.
            **kwargs: Additional arguments (e.g., tool_call_id for tool messages).

        Raises:
            ValueError: If the role is unsupported.
        """
        message_map = {
            "user": Message.user_message,
            "system": Message.system_message,
            "assistant": Message.assistant_message,
            "tool": lambda content, **kw: Message.tool_message(content, **kw),
        }

        if role not in message_map:
            raise ValueError(f"Unsupported message role: {role}")

        # Create message with appropriate parameters based on role
        call_kw: Dict[str, Any] = {"base64_image": base64_image}
        if role == "tool":
            call_kw.update(kwargs)
        elif role == "assistant" and "reasoning_content" in kwargs:
            call_kw["reasoning_content"] = kwargs["reasoning_content"]
        message = message_map[role](content, **call_kw)
        self.append_message(message)
        self._emit_event(
            "agent_memory_add",
            {
                "role": role,
                "content_len": len(content or ""),
                "has_image": bool(base64_image),
            },
        )

    async def run(
        self,
        request: Optional[str] = None,
        *,
        use_long_memory: bool = False,
        only_save_last_result_to_short: bool = False,
    ) -> str:
        """Execute the agent's main loop asynchronously.

        Args:
            request: Optional initial user request to process.
            use_long_memory: If True, inject stored long_memory into short_memory.

        Returns:
            A string summarizing the execution results.

        Raises:
            RuntimeError: If the agent is not in IDLE state at start.
        """
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")
        
        # reset run_id, keep run_id for awaiting_human
        if not self.awaiting_human:
            self.run_id = uuid.uuid4().hex[:12]
            self.short_memory.run_id = self.run_id
            self.short_memory.flushed_this_run = False
        
        # Prune leftover short_memory before a new run (multi-turn runtime loops).
        if self.short_memory.messages:
            try:
                if await self.short_memory.prune(llm=self.llm):
                    logger.info(
                        "Short memory pruned before run (messages=%s strategy=%s)",
                        len(self.short_memory.messages),
                        self.short_memory.cleanup_strategy.value,
                    )
            except Exception as exc:
                logger.warning("Short memory prune before run failed: %s", exc)

        # record request immediately.
        self.update_memory("user", request)
        self.working_memory.clear_without_system_messages()

        # New run begins; if we were previously paused, we are now resuming.
        self.awaiting_human = False
        self.last_request = request
        self.last_result = None

        results: List[str] = []
        output: Optional[str] = None
        self._emit_event(
            "agent_run_start",
            {
                "name": self.name,
                "max_steps": self.max_steps,
                "request_len": len(request or ""),
                "run_id": self.run_id,
            },
        )
        try:
            async with self.state_context(AgentState.RUNNING):
                while (
                    self.current_step < self.max_steps and self.state != AgentState.FINISHED
                ):
                    self.current_step += 1
                    logger.info(f"Executing step {self.current_step}/{self.max_steps}")
                    self._emit_event(
                        "agent_step_start",
                        {
                            "name": self.name,
                            "current_step": self.current_step,
                            "max_steps": self.max_steps,
                        },
                    )

                    # Memory flush: evict oldest messages to sqlite when over limit.
                    try:
                        result = self.short_memory.flush(run_id=self.run_id)
                        if result.flushed:
                            logger.info(
                                "Short memory flushed run_id=%s flushed=%s kept=%s",
                                self.run_id,
                                result.flushed_count,
                                result.kept_count,
                            )
                    except Exception as e:
                        logger.warning("Short memory flush failed: %s", e)

                    step_result = await self.step()
                    self._emit_event(
                        "agent_step_end",
                        {
                            "name": self.name,
                            "current_step": self.current_step,
                            "max_steps": self.max_steps,
                            "result_len": len(step_result or ""),
                        },
                    )

                    # Check for stuck state
                    if self.is_stuck():
                        self.handle_stuck_state()

                    results.append(f"Step {self.current_step}: {step_result}")

                if self.current_step >= self.max_steps:
                    self.current_step = 0
                    self.state = AgentState.IDLE
                    results.append(
                        f"Terminated: Reached max steps ({self.max_steps})"
                    )
            # self.last_result = step_result if step_result else "No last result as output"
            output = "\n".join(results) if results else "No steps executed"

            self._emit_event(
                "agent_run_end",
                {
                    "name": self.name,
                    "current_step": self.current_step,
                    "state": str(self.state),
                    "output_len": len(output),
                    "run_id": self.run_id,
                },
            )

            # Select the last result from the short memory
            if self.summarize_short_memory:
                from weakagent.llm.summarize import summarize_short_memory as _summarize_short
                summary_msg = await _summarize_short(
                    self.llm, list(self.short_memory.messages)
                )
                output = summary_msg.content or ""
                return output
            if self.only_last_result:
                output = self.last_result
                return output
            return output
        
        finally:
            # Snapshot: merge flushed segments + RAM, or save full short_memory.
            try:
                self.short_memory.finalize_run_memory(run_id=self.run_id)
            except Exception:
                logger.exception("Short memory finalize failed")

            try:
                if not self.awaiting_human:
                    if self.only_save_last_result_to_short:
                        self.short_memory.clear_unitl_last_user_message()
                        self.short_memory.add_message(Message.assistant_message(output))

                    if self.summarize_working_memory:
                        threading.Thread(
                            target=self.working_memory.summarize_and_save,
                            args=(self.run_id,),
                        ).start()

            except Exception:
                logger.exception("Failed after run")

    @abstractmethod
    async def step(self) -> str:
        """Execute a single step in the agent's workflow.

        Must be implemented by subclasses to define specific behavior.
        """

    def handle_stuck_state(self):
        """Handle stuck state by adding a prompt to change strategy"""
        stuck_prompt = "\
        Observed duplicate responses. Consider new strategies and avoid repeating ineffective paths already attempted."
        self.next_step_prompt = f"{stuck_prompt}\n{self.next_step_prompt}"
        logger.warning(f"Agent detected stuck state. Added prompt: {stuck_prompt}")

    def is_stuck(self) -> bool:
        """Check if the agent is stuck in a loop by detecting duplicate content"""
        if len(self.short_memory.messages) < 2:
            return False

        last_message = self.short_memory.messages[-1]
        if not last_message.content:
            return False

        # Count identical content occurrences
        duplicate_count = sum(
            1
            for msg in reversed(self.short_memory.messages[:-1])
            if msg.role == "assistant" and msg.content == last_message.content
        )

        return duplicate_count >= self.duplicate_threshold

    @property
    def messages(self) -> List[Message]:
        """Retrieve a list of messages from the agent's memory."""
        return self.short_memory.messages

    @messages.setter
    def messages(self, value: List[Message]):
        """Set the list of messages in the agent's memory."""
        self.short_memory.messages = value
