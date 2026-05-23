from typing import List, Optional, Union
from enum import Enum
from pydantic import Field

from weakagent.agent.toolcall import ToolCallAgent
from weakagent.utils.exceptions import TokenLimitExceeded
from weakagent.utils.logger import get_logger
from weakagent.prompt.brief_react import THINK_NEXT_STEP_PROMPT, THINK_SYSTEM_PROMPT, ACT_NEXT_STEP_PROMPT, ACT_SYSTEM_PROMPT
from weakagent.llm.llm import _extract_reasoning_content
from weakagent.schemas.tool import TOOL_CHOICE_TYPE, ToolCall, ToolChoice
from weakagent.schemas.agent import AgentState
from weakagent.schemas.message import ROLE_TYPE, Message, Role
from weakagent.tools import ToolCollection, CreateChatCompletion, Terminate, AskHumanTool


class BriefMessageProccess(str, Enum):
    """Brief message proccess type: how to process the message in brief_react."""
    USER = "user" # process the message as user message
    ASSISTANT = "assistant" # process the message as assistant message
    NONE = "none" # do not process the message, not add to memory

TOOL_CALL_REQUIRED = "Tool calls required but none provided"
logger = get_logger(__name__)

class BriefReActAgent(ToolCallAgent):
    name: str = "react"
    description: str = "an agent that git a brieft think, selcet tools to execute tool calls using react."
    available_tools: ToolCollection = ToolCollection(
        CreateChatCompletion(), Terminate(), AskHumanTool()
    )
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name, AskHumanTool().name])
    
    think_system_prompt: str = THINK_SYSTEM_PROMPT
    think_next_step_prompt: str = THINK_NEXT_STEP_PROMPT
    act_system_prompt: str = ACT_SYSTEM_PROMPT
    act_next_step_prompt: str = ACT_NEXT_STEP_PROMPT

    brief_hink_message_proccess : BriefMessageProccess = Field(default=BriefMessageProccess.NONE)

    async def think(self) -> bool:

        # get a brief think
        request_messages = list(self.messages)
        if self.think_next_step_prompt:
            request_messages.append(Message.user_message(self.think_next_step_prompt))
        
        try:
            content = await self.llm.ask(
                messages=request_messages,
                system_msgs=[Message.system_message(self.think_system_prompt)],
                temperature=0.0,
                verbose=self.verbose,
            )

            logger.info(f"Brief think content: {content}")
            logger.info(f"Brief think reasoning content: {self.llm.last_reasoning_content}")
            if self.brief_hink_message_proccess == BriefMessageProccess.USER:
                self.update_memory(Role.USER, content, reasoning_content=self.llm.last_reasoning_content)
            elif self.brief_hink_message_proccess == BriefMessageProccess.ASSISTANT:
                self.update_memory(Role.ASSISTANT, content, reasoning_content=self.llm.last_reasoning_content)
            elif self.brief_hink_message_proccess == BriefMessageProccess.NONE:
                pass
            else:
                raise ValueError(f"Invalid brief_hink_message_proccess: {self.brief_hink_message_proccess}")

            self.act_next_step_prompt = content
        except ValueError:
            raise
        except Exception as e:
            # Check if this is a RetryError containing TokenLimitExceeded
            if hasattr(e, "__cause__") and isinstance(e.__cause__, TokenLimitExceeded):
                token_limit_error = e.__cause__
                logger.error(
                    f"🚨 Token limit error (from RetryError): {token_limit_error}"
                )
                self.update_memory(
                    "assistant",
                    f"Maximum token limit reached, cannot continue execution: {str(token_limit_error)}",
                )
                self.state = AgentState.FINISHED
                return False
            raise

        # request_messages just use to llm ask, but will not add to memory
        request_messages = list(self.messages)
        if self.act_next_step_prompt:
            request_messages.append(Message.user_message(self.act_next_step_prompt))

        try:
            # Get response with tool options
            response = await self.llm.ask_tool(
                messages=request_messages,
                system_msgs=(
                    [Message.system_message(self.with_skills_prompt(self.act_system_prompt))]
                    if self.act_system_prompt
                    else None
                ),
                tools=self.available_tools.to_params(),
                tool_choice=self.tool_choices,
                verbose=self.verbose,
            )
        except ValueError:
            raise
        except Exception as e:
            # Check if this is a RetryError containing TokenLimitExceeded
            if hasattr(e, "__cause__") and isinstance(e.__cause__, TokenLimitExceeded):
                token_limit_error = e.__cause__
                logger.error(
                    f"🚨 Token limit error (from RetryError): {token_limit_error}"
                )
                self.update_memory(
                    "assistant",
                    f"Maximum token limit reached, cannot continue execution: {str(token_limit_error)}",
                )
                self.state = AgentState.FINISHED
                return False
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