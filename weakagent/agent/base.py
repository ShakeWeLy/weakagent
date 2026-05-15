from abc import ABC, abstractmethod
import asyncio
from contextlib import asynccontextmanager
import inspect
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from weakagent.llm import LLM
from weakagent.utils.logger import get_logger
from weakagent.schemas.message import ROLE_TYPE, Message
from weakagent.memory.conversation import ConversationMemory
from weakagent.memory.short import ShortMemory
from weakagent.memory.runtime_memory import RuntimeMemory
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
    short_memory: ShortMemory = Field(
        default_factory=ShortMemory, description="Per-run agent working context"
    )
    conversation: Optional[ConversationMemory] = Field(
        default=None, description="Persistent conversation store"
    )
    runtime_memory: RuntimeMemory = Field(
        default_factory=RuntimeMemory,
        description="Keeps only request + final output of each run; not cleared after run",
    )
    awaiting_human: bool = Field(
        default=False,
        description="If true, this run is paused awaiting user input; do not persist last_result.",
    )
    # Runtime-level persistence helpers (used by AgentRuntime to write RuntimeMemory).
    last_request: Optional[str] = Field(default=None)
    last_result: Optional[str] = Field(default=None)
    run_id: Optional[str] = Field(default=None)
    state: AgentState = Field(
        default=AgentState.IDLE, description="Current agent state"
    )
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
    only_last_result: bool = Field(default="", description="If only retutn last output result from agent")

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

    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        """Initialize agent with default settings if not provided."""
        if self.llm is None or not isinstance(self.llm, LLM):
            self.llm = LLM(config_name=self.name.lower())
        if not isinstance(self.short_memory, ShortMemory):
            self.short_memory = ShortMemory()
        if self.conversation is None:
            self.conversation = ConversationMemory(
                session_id=f"sess_{self.name}_{uuid.uuid4().hex[:8]}",
                agent_type=self.name,
                title=self.description or self.name,
            )
        return self

    def append_message(self, message: Message, *, extra: Optional[Dict[str, Any]] = None) -> None:
        """Append message to memory and conversation storage."""
        self.short_memory.add_message(message)
        if self.conversation is None:
            return
        persist_extra = {
            "agent": self.name,
            "state": str(self.state),
            "current_step": self.current_step,
        }
        if extra:
            persist_extra.update(extra)
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
        kwargs = {"base64_image": base64_image, **(kwargs if role == "tool" else {})}
        message = message_map[role](content, **kwargs)
        self.append_message(message)
        self._emit_event(
            "agent_memory_add",
            {
                "role": role,
                "content_len": len(content or ""),
                "has_image": bool(base64_image),
            },
        )

    async def run(self, request: Optional[str] = None) -> str:
        """Execute the agent's main loop asynchronously.

        Args:
            request: Optional initial user request to process.

        Returns:
            A string summarizing the execution results.

        Raises:
            RuntimeError: If the agent is not in IDLE state at start.
        """
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        # RuntimeMemory: record request immediately (not cleared after run).
        self.runtime_memory.add_request(request)

        # Each run uses an independent short_memory context.
        if not self.awaiting_human:
            self.short_memory.clear()
            # Load runtime_memory history into this run's short_memory context.
            if self.runtime_memory.messages:
                self.short_memory.add_messages(list(self.runtime_memory.messages))
        # If awaiting human, add request to memory and keep the lasted short_memory messages(no clear).
        else:
            self.update_memory("user", request)

        # New run begins; if we were previously paused, we are now resuming.
        self.awaiting_human = False
        self.last_request = request
        self.last_result = None

        results: List[str] = []
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        self.run_id = run_id
        self._emit_event(
            "agent_run_start",
            {
                "name": self.name,
                "max_steps": self.max_steps,
                "request_len": len(request or ""),
                "run_id": run_id,
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

                    # Memory hygiene before each step (before any potential LLM call in step()).
                    cleanup = getattr(self.short_memory, "cleanup_if_needed", None)
                    if callable(cleanup):
                        try:
                            await cleanup(llm=self.llm)
                        except Exception as e:
                            logger.warning(f"Memory cleanup failed: {e}")

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
            self.last_result = step_result if step_result else "No last result as output"
            output = "\n".join(results) if results else "No steps executed"

            self._emit_event(
                "agent_run_end",
                {
                    "name": self.name,
                    "current_step": self.current_step,
                    "state": str(self.state),
                    "output_len": len(output),
                    "run_id": run_id,
                },
            )

            if self.conversation is not None:
                try:
                    await self.conversation.write_session_summary(
                        run_id=run_id,
                        status=str(self.state),
                        llm=LLM(config_name="fast"),
                        extra={"stream": False},
                    )
                except Exception:
                    logger.exception("Failed to write session summary")
            # RuntimeMemory: append final output after run completes.
            try:
                self.runtime_memory.add_last_result(self.last_result)
            except Exception:
                logger.exception("Failed to write runtime_memory last_result")
            
            if self.only_last_result:
                return self.last_result
            return output
        finally:
            # After run, clear the per-run context to keep runs independent.
            try:
                if not self.awaiting_human:
                    self.short_memory.clear()
            except Exception:
                logger.exception("Failed to clear short_memory after run")

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
