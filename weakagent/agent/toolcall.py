import asyncio
import json
import time
from typing import Any, List, Optional, Union, overload

from pydantic import Field

from weakagent.agent.react import ReActAgent
from weakagent.agent.llm_errors import handle_llm_think_error
from weakagent.llm.llm import _extract_reasoning_content
from weakagent.utils.logger import get_logger
from weakagent.prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from weakagent.schemas.tool import TOOL_CHOICE_TYPE, ToolCall, ToolChoice
from weakagent.schemas.agent import AgentState
from weakagent.schemas.message import Message
from weakagent.tools import ToolCollection, Terminate, WebSearch
from weakagent.tools.special_tool.ask_human import AskHumanTool
from weakagent.tools.base import BaseTool
from weakagent.tools.tool import HotReloadTool, ListToolsTool

TOOL_CALL_REQUIRED = "Tool calls required but none provided"
logger = get_logger(__name__)


class ToolCallAgent(ReActAgent):
    """Base agent class for handling tool/function calls with enhanced abstraction"""

    name: str = "toolcall"
    description: str = "an agent that can execute tool calls."

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    available_tools: ToolCollection = ToolCollection(
        WebSearch(),
        AskHumanTool(),
        Terminate(),
        ListToolsTool(),
        HotReloadTool(),
    )
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore
    special_tool_names: List[str] = Field(
        default_factory=lambda: [AskHumanTool().name, Terminate().name]
    )

    tool_calls: List[ToolCall] = Field(default_factory=list)
    _current_base64_image: Optional[str] = None

    max_steps: int = 30
    max_observe: Optional[Union[int, bool]] = None # max observe in tool execute result.

    async def think(self) -> bool:
        """Process current state and decide next actions using tools"""
        # 把 NEXT_STEP_PROMPT 改成“仅作为本轮推理的临时消息传给 LLM，不写入 memory”，这样终端里的 Memory Trace 就不会再出现那条伪造的 user 消息。
        request_messages = list(self.messages)
        if self.next_step_prompt:
            request_messages.append(Message.user_message(self.next_step_prompt.format(task=self.request)))

        try:
            # Get response with tool options
            response = await self.llm.ask_tool(
                messages=request_messages,
                system_msgs= self.system_messages,
                tools=self.available_tools.to_params(),
                tool_choice=self.tool_choices,
                verbose=self.verbose
            )
        except ValueError:
            raise
        except Exception as e:
            handled = handle_llm_think_error(self, e)
            if handled is not None:
                return handled
            raise

        self.tool_calls = tool_calls = (
            response.tool_calls if response and response.tool_calls else []
        )
        content = response.content if response and response.content else ""
        reasoning = _extract_reasoning_content(response)

        # Log response info
        logger.info(f"✨ {self.name}'s thoughts: {content}")
        logger.info(
            f"🛠️ {self.name} selected {len(tool_calls) if tool_calls else 0} tools to use"
        )
        if tool_calls:
            logger.info(
                f"🧰 Tools being prepared: {[call.function.name for call in tool_calls]}"
            )
            logger.info(f"🔧 Tool arguments: {tool_calls[0].function.arguments}")

        try:
            if response is None:
                raise RuntimeError("No response received from the LLM")

            # Handle different tool_choices modes
            if self.tool_choices == ToolChoice.NONE:
                if tool_calls:
                    logger.warning(
                        f"🤔 Hmm, {self.name} tried to use tools when they weren't available!"
                    )
                if content:
                    self.update_memory(
                        "assistant",
                        content,
                        reasoning_content=reasoning,
                    )
                    return True
                return False

            # Create and add assistant message
            assistant_msg = (
                Message.from_tool_calls(
                    content=content,
                    tool_calls=self.tool_calls,
                    reasoning_content=reasoning,
                )
                if self.tool_calls
                else Message.assistant_message(content, reasoning_content=reasoning)
            )
            self.append_message(assistant_msg)

            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True  # Will be handled in act()

            # For 'auto' mode, continue with content if no commands but content exists
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return bool(content)

            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"🚨 Oops! The {self.name}'s thinking process hit a snag: {e}")
            self.update_memory(
                "assistant",
                f"Error encountered while processing: {str(e)}",
            )
            return False

    async def act(self) -> str:
        """Execute tool calls and handle their results"""
        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                raise ValueError(TOOL_CALL_REQUIRED)
            # Return last message content and change agent state to finished
            if self.messages[-1].content:
                final_content = self.messages[-1].content
                self.update_memory(
                    "assistant",
                    final_content,
                )
                self.last_result = final_content
                self.state = AgentState.FINISHED
                logger.info(
                    "🏁 Agent completed the task (no tool calls, final response in content)."
                )
                return final_content
            else:
                return "No content or commands to execute"

        results = []
        for command in self.tool_calls:
            # Reset base64_image for each tool call
            self._current_base64_image = None

            result = await self.execute_tool(command)
            _obs_prefix = f"Observed output of cmd `{command.function.name}` executed:\n"
            self.last_result = result.removeprefix(_obs_prefix)

            if self.max_observe:
                result = result[: self.max_observe]

            logger.info(
                f"🎯 Tool '{command.function.name}' completed its mission! Result: {result}"
            )

            # Add tool response to memory
            self.update_memory(
                "tool",
                result,
                tool_call_id=command.id,
                name=command.function.name,
                base64_image=self._current_base64_image,
            )
            results.append(result)

        return "\n\n".join(results)

    async def execute_tool(self, command: ToolCall) -> str:
        """Execute a single tool call with robust error handling"""
        if not command or not command.function or not command.function.name:
            return "Error: Invalid command format"

        name = command.function.name
        if name not in self.available_tools.tool_map:
            return f"Error: Unknown tool '{name}'"

        try:
            # Parse arguments
            args = json.loads(command.function.arguments or "{}")

            # Execute the tool
            logger.info(f"🔧 Activating tool: '{name}'...")
            self._emit_event(
                "agent_tool_call_start",
                {
                    "tool_name": name,
                    "tool_call_id": getattr(command, "id", None),
                    "args": args,
                    "current_step": getattr(self, "current_step", 0),
                },
            )
            t0 = time.perf_counter()
            tool = self.available_tools.get_tool(name)
            execute_for_agent = (
                getattr(tool, "execute_for_agent", None) if tool is not None else None
            )
            if callable(execute_for_agent):
                result = await execute_for_agent(self, **args)
            else:
                result = await self.available_tools.execute(
                    name=name, tool_input=args
                )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Handle special tools
            await self._handle_special_tool(name=name, result=result)

            # Check if result is a ToolResult with base64_image
            if hasattr(result, "base64_image") and result.base64_image:
                # Store the base64_image for later use in tool_message
                self._current_base64_image = result.base64_image

            # Format result for display (standard case)
            observation = (
                f"Observed output of cmd `{name}` executed:\n{result.output}"
                if result.success and result.output
                else f"Cmd `{name}` completed with no output: {result.error}"
            )

            self._emit_event(
                "agent_tool_call_end",
                {
                    "tool_name": name,
                    "tool_call_id": getattr(command, "id", None),
                    "current_step": getattr(self, "current_step", 0),
                    "duration_ms": elapsed_ms,
                    "success": bool(getattr(result, "success", False)),
                    "output_len": len((getattr(result, "output", "") or "")),
                    "error": getattr(result, "error", None),
                },
            )
            return observation
        except json.JSONDecodeError:
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
            logger.error(
                f"📝 Oops! The arguments for '{name}' don't make sense - invalid JSON, arguments:{command.function.arguments}"
            )
            self._emit_event(
                "agent_tool_call_end",
                {
                    "tool_name": name,
                    "tool_call_id": getattr(command, "id", None),
                    "current_step": getattr(self, "current_step", 0),
                    "success": False,
                    "error": "Invalid JSON arguments",
                },
            )
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
            logger.exception(error_msg)
            self._emit_event(
                "agent_tool_call_end",
                {
                    "tool_name": name,
                    "tool_call_id": getattr(command, "id", None),
                    "current_step": getattr(self, "current_step", 0),
                    "success": False,
                    "error": str(e),
                },
            )
            return f"Error: {error_msg}"

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """Handle special tool execution and state changes"""
        if not self._is_special_tool(name):
            return

        # AskHuman pauses the run: do not treat current output as final.
        try:
            if getattr(result, "data", None) and result.data.get("await_human"):
                setattr(self, "awaiting_human", True)
        except Exception:
            pass

        if self._should_finish_execution(name=name, result=result, **kwargs):
            # Set agent state to finished
            logger.info(f"🏁 Special tool '{name}' has completed the task!")
            self.state = AgentState.FINISHED

    @overload
    async def add_tool_dynamically(self, tool: str) -> BaseTool: ...

    @overload
    async def add_tool_dynamically(self, tool: BaseTool) -> BaseTool: ...

    async def add_tool_dynamically(self, tool: Union[BaseTool, str]) -> BaseTool:
        """Mount a tool at runtime (instance or built-in registry name).

        Args:
            tool: A :class:`BaseTool` instance, or a built-in tool name from
                :meth:`ToolCollection.add_tool_by_name`.

        Returns:
            The mounted tool instance.

        Raises:
            ValueError: When *tool* is a name that cannot be resolved or instantiated.
        """
        if isinstance(tool, str):
            mounted = self.available_tools.add_tool_by_name(tool)
            if mounted is None:
                raise ValueError(f"Unknown or failed to load built-in tool: {tool!r}")
            logger.info("Dynamically mounted tool %s", mounted.name)
            return mounted

        if tool.name in self.available_tools.tool_map:
            logger.info("Tool %s already mounted", tool.name)
            return self.available_tools.tool_map[tool.name]

        self.available_tools.add_tool(tool)
        logger.info("Dynamically mounted tool %s", tool.name)
        return tool

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        """Determine if tool execution should finish the agent"""
        return True

    def _is_special_tool(self, name: str) -> bool:
        """Check if tool name is in special tools list"""
        return name.lower() in [n.lower() for n in self.special_tool_names]

    async def cleanup(self):
        """Clean up resources used by the agent's tools."""
        logger.info(f"🧹 Cleaning up resources for agent '{self.name}'...")
        for tool_name, tool_instance in self.available_tools.tool_map.items():
            if hasattr(tool_instance, "cleanup") and asyncio.iscoroutinefunction(
                tool_instance.cleanup
            ):
                try:
                    logger.debug(f"🧼 Cleaning up tool: {tool_name}")
                    await tool_instance.cleanup()
                except Exception as e:
                    logger.error(
                        f"🚨 Error cleaning up tool '{tool_name}': {e}", exc_info=True
                    )
        logger.info(f"✨ Cleanup complete for agent '{self.name}'.")

    async def run(
        self,
        request: Optional[str] = None,
    ) -> str:
        """Run the agent with cleanup when done."""
        try:
            return await super().run(request)
        finally:
            await self.cleanup()
