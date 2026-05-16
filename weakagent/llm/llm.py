"""
LLM 客户端：仅使用 OpenAI Python SDK（``AsyncOpenAI``），

支持官方 API 与任意 OpenAI 兼容的 ``base_url``。显式构造 ``LLM(...)``；

或经 :class:`weakagent.llm.factory.LLMFactory` 创建；不做进程内单例。
"""
import asyncio
import inspect
import json
import time
import uuid
from typing import Any, Awaitable, Callable, List, Optional, Union

import tiktoken
from openai import (
    APIError,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from weakagent.config.settings import LLMSettings, config
from .token_counter import TokenCounter
from weakagent.utils.exceptions import ModelCapabilityError, TokenLimitExceeded
from weakagent.utils.logger import get_logger  # Assuming a logger is set up in your app
from weakagent.utils.verbose import get_real_caller
from weakagent.schemas.message import (
    ROLE_VALUES,
    Message,
)
from weakagent.schemas.tool import (
    ToolChoice,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
)

logger = get_logger(__name__)

EventCallback = Callable[[dict], Union[Any, Awaitable[Any]]]


def _extract_reasoning_content(obj: Any) -> Optional[str]:
    """Read provider-specific thinking/reasoning text from a completion message or delta."""
    if obj is None:
        return None
    rc = getattr(obj, "reasoning_content", None)
    if rc is None:
        extra = getattr(obj, "model_extra", None)
        if isinstance(extra, dict):
            rc = extra.get("reasoning_content")
    if rc is None:
        return None
    return rc if isinstance(rc, str) else str(rc)


class LLM:
    def __init__(
        self,
        config_name: str = "default",
        llm_config: Optional[LLMSettings] = None,
        *,
        on_event: Optional[EventCallback] = None,
    ):
        llm_config = llm_config or config.llm
        llm_config = llm_config.get(config_name, llm_config["default"])
        self.model = llm_config.model
        self.max_tokens = llm_config.max_tokens
        self.temperature = llm_config.temperature
        self.api_key = llm_config.api_key
        self.base_url = llm_config.base_url
        self.supports_images = llm_config.supports_images
        self.use_max_completion_tokens = llm_config.use_max_completion_tokens
        self.context_window = getattr(llm_config, "context_window", None)
        self.reserve_completion_tokens = getattr(
            llm_config, "reserve_completion_tokens", None
        )

        # Add token counting related attributes
        self.total_input_tokens = 0
        self.total_completion_tokens = 0
        self.max_input_tokens = (
            llm_config.max_input_tokens
            if hasattr(llm_config, "max_input_tokens")
            else None
        )

        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # If the model is not in tiktoken's presets, use cl100k_base as default
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        self.token_counter = TokenCounter(self.tokenizer)
        self.on_event = on_event
        # Set by ask / ask_tool / ask_tool_stream when the provider returns thinking content.
        self._last_reasoning_content: Optional[str] = None

    @property
    def last_reasoning_content(self) -> Optional[str]:
        """Reasoning/thinking text from the last completion, if any (OpenAI-compat extensions)."""
        return self._last_reasoning_content

    def _emit_event(
        self,
        event_type: str,
        data: Optional[dict] = None,
        *,
        on_event: Optional[EventCallback] = None,
    ) -> None:
        """Emit an event to `on_event` callback (sync/async; isolated failures)."""
        cb = on_event if on_event is not None else self.on_event
        if cb is None:
            return
        try:
            result = cb(
                {
                    "type": event_type,
                    "timestamp": time.time(),
                    "data": data or {},
                }
            )
            if inspect.isawaitable(result):
                task = asyncio.create_task(result)

                def _on_event_done(t: asyncio.Task) -> None:
                    try:
                        exc = t.exception()
                    except asyncio.CancelledError:
                        return
                    except Exception:
                        logger.exception("Unexpected error while checking on_event task")
                        return
                    if exc is not None:
                        logger.exception("on_event async callback failed", exc_info=exc)

                task.add_done_callback(_on_event_done)
        except Exception as e:
            logger.error(f"Event callback error: {e}")

    def count_tokens(self, text: str) -> int:
        """Calculate the number of tokens in a text"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, messages: List[dict]) -> int:
        return self.token_counter.count_message_tokens(messages)

    def update_token_count(self, input_tokens: int, completion_tokens: int = 0) -> None:
        """Update token counts"""
        # Only track tokens if max_input_tokens is set
        self.total_input_tokens += input_tokens
        self.total_completion_tokens += completion_tokens
        logger.info(
            f"Token usage: Input={input_tokens}, Completion={completion_tokens}, "
            f"Cumulative Input={self.total_input_tokens}, Cumulative Completion={self.total_completion_tokens}, "
            f"Total={input_tokens + completion_tokens}, Cumulative Total={self.total_input_tokens + self.total_completion_tokens}"
        )

    def check_token_limit(self, input_tokens: int) -> bool:
        """Check if token limits are exceeded"""
        if self.max_input_tokens is not None:
            return (self.total_input_tokens + input_tokens) <= self.max_input_tokens
        # If max_input_tokens is not set, always return True
        return True

    def get_limit_error_message(self, input_tokens: int) -> str:
        """Generate error message for token limit exceeded"""
        if (
            self.max_input_tokens is not None
            and (self.total_input_tokens + input_tokens) > self.max_input_tokens
        ):
            return f"Request may exceed input token limit (Current: {self.total_input_tokens}, Needed: {input_tokens}, Max: {self.max_input_tokens})"

        return "Token limit exceeded"

    @staticmethod
    def format_messages(
        messages: List[Union[dict, Message]], supports_images: bool = False
    ) -> List[dict]:
        """
        Format messages for LLM by converting them to OpenAI message format.

        Args:
            messages: List of messages that can be either dict or Message objects
            supports_images: Flag indicating if the target model supports image inputs

        Returns:
            List[dict]: List of formatted messages in OpenAI format

        Raises:
            ValueError: If messages are invalid or missing required fields
            TypeError: If unsupported message types are provided

        Examples:
            >>> msgs = [
            ...     Message.system_message("You are a helpful assistant"),
            ...     {"role": "user", "content": "Hello"},
            ...     Message.user_message("How are you?")
            ... ]
            >>> formatted = LLM.format_messages(msgs)
        """
        formatted_messages = []

        for message in messages:
            # Convert Message objects to dictionaries
            if isinstance(message, Message):
                message = message.to_dict()

            if isinstance(message, dict):
                # Avoid mutating caller-owned dicts.
                message = dict(message)

                # If message is a dict, ensure it has required fields
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")

                # Process base64 images if present and model supports images
                if supports_images and message.get("base64_image"):
                    # Initialize or convert content to appropriate format
                    if not message.get("content"):
                        message["content"] = []
                    elif isinstance(message["content"], str):
                        message["content"] = [
                            {"type": "text", "text": message["content"]}
                        ]
                    elif isinstance(message["content"], list):
                        # Convert string items to proper text objects
                        message["content"] = [
                            (
                                {"type": "text", "text": item}
                                if isinstance(item, str)
                                else item
                            )
                            for item in message["content"]
                        ]

                    # Add the image to content (pass through if already a data URL)
                    _img = message["base64_image"]
                    if (
                        isinstance(_img, str)
                        and _img.startswith("data:")
                        and "base64," in _img
                    ):
                        _image_url = _img
                    else:
                        _image_url = f"data:image/jpeg;base64,{_img}"

                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _image_url,
                            },
                        }
                    )

                    # Remove the base64_image field
                    del message["base64_image"]
                # If model doesn't support images but message has base64_image, handle gracefully
                elif not supports_images and message.get("base64_image"):
                    # Just remove the base64_image field and keep the text content
                    if not message.get("content"):
                        logger.warning(
                            "format_messages: dropping image-only message (model doesn't support images). "
                            "role=%s keys=%s",
                            message.get("role"),
                            list(message.keys()),
                        )
                    del message["base64_image"]

                if (
                    "content" in message
                    or "tool_calls" in message
                    or message.get("reasoning_content")
                ):
                    formatted_messages.append(message)
                else:
                    # This is a common source of "why did the message disappear?"
                    logger.warning(
                        "format_messages: skipping message because neither 'content' nor 'tool_calls' present. "
                        "role=%s keys=%s",
                        message.get("role"),
                        list(message.keys()),
                    )
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # Validate all messages have required fields
        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"Invalid role: {msg['role']}")

        return formatted_messages

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
        verbose: bool = False,
        on_event: Optional[EventCallback] = None,
    ) -> str:
        """
        Send a prompt to the LLM and get the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response
            verbose (bool): Whether to print the messages.
            on_event: Optional event callback (sync/async). If not provided, uses `self.on_event`.
                Intended for streaming side-effects (front-end push, TTS, etc.). Failures are isolated.
        Returns:
            str: The generated response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors

        Note:
            多模态由配置项 ``supports_images`` 控制，不再按模型名列表判断。
        """
        try:
            supports_images = self.supports_images
            self._last_reasoning_content = None

            # Format system and user messages with image support check
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # Calculate input token count
            input_tokens = self.count_message_tokens(messages)

            # Check if token limits are exceeded
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # Raise a special exception that won't be retried
                raise TokenLimitExceeded(error_message)

            params = {
                "model": self.model,
                "messages": messages,
            }

            if self.use_max_completion_tokens:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            if not stream:
                # Non-streaming request
                response = await self.client.chat.completions.create(
                    **params, stream=False
                )

                msg = response.choices[0].message if response.choices else None
                if not msg or not (msg.content or _extract_reasoning_content(msg)):
                    raise ValueError("Empty or invalid response from LLM")

                # Update token counts
                self.update_token_count(
                    response.usage.prompt_tokens, response.usage.completion_tokens
                )

                self._last_reasoning_content = _extract_reasoning_content(msg)
                return msg.content or ""

            # Streaming request, For streaming, update estimated token count before making the request
            self.update_token_count(input_tokens)
            
            if verbose:
                logger.info(f"[VERBOSE] caller={get_real_caller()} model={self.model} messages={messages}")
            
            response = await self.client.chat.completions.create(**params, stream=True)

            collected_messages = []
            print("-"*20)
            completion_text = ""
            reasoning_text = ""

            self._emit_event(
                "llm_stream_start",
                {
                    "model": self.model,
                    "input_tokens": input_tokens,
                    "stream": True,
                },
                on_event=on_event,
            )

            async for chunk in response:
                # Some OpenAI-compatible providers may yield chunks with empty `choices`
                # (or with no `delta.content`). Guard to avoid `IndexError`.
                if not getattr(chunk, "choices", None):
                    continue

                choice0 = chunk.choices[0]
                delta = getattr(choice0, "delta", None)
                if delta is not None:
                    rchunk = _extract_reasoning_content(delta)
                    if rchunk:
                        reasoning_text += rchunk

                chunk_message = (
                    (getattr(delta, "content", None) or "") if delta is not None else ""
                )

                if not chunk_message:
                    continue

                collected_messages.append(chunk_message)
                completion_text += chunk_message
                print(chunk_message, end="", flush=True)
                self._emit_event(
                    "llm_stream_chunk",
                    {
                        "text": chunk_message,
                        "accumulated_text_len": len(completion_text),
                    },
                    on_event=on_event,
                )
            if not completion_text:
                print("No print content from LLM")
            print()  # Newline after streaming
            print("-"*20)

            full_response = "".join(collected_messages).strip()
            if not full_response:
                print("No print content from LLM")
                raise ValueError("Empty response from streaming LLM")

            self._last_reasoning_content = reasoning_text or None

            # estimate completion tokens for streaming response
            completion_tokens = self.count_tokens(completion_text)
            logger.info(
                f"[LLM] {self.model} estimated completion tokens for streaming response: {completion_tokens}"
            )
            self.total_completion_tokens += completion_tokens

            self._emit_event(
                "llm_stream_end",
                {
                    "model": self.model,
                    "estimated_completion_tokens": completion_tokens,
                    "output_text_len": len(full_response),
                },
                on_event=on_event,
            )
            return full_response

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except ValueError:
            logger.exception(f"Validation error")
            self._emit_event(
                "llm_error",
                {"kind": "ValueError"},
                on_event=on_event,
            )
            raise
        except OpenAIError as oe:
            logger.exception(f"OpenAI API error")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            self._emit_event(
                "llm_error",
                {"kind": "OpenAIError", "error_type": oe.__class__.__name__},
                on_event=on_event,
            )
            raise
        except Exception:
            logger.exception(f"Unexpected error in ask")
            self._emit_event(
                "llm_error",
                {"kind": "Exception"},
                on_event=on_event,
            )
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask_with_images(
        self,
        messages: List[Union[dict, Message]],
        images: List[Union[str, dict]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        on_event: Optional[EventCallback] = None,
    ) -> str:
        """
        Send a prompt with images to the LLM and get the response.

        Args:
            messages: List of conversation messages
            images: List of image URLs or image data dictionaries
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response
            on_event: Optional event callback (sync/async). If not provided, uses `self.on_event`.
                Intended for streaming side-effects (front-end push, TTS, etc.). Failures are isolated.

        Returns:
            str: The generated response

        Raises:
            ModelCapabilityError: 若配置中 ``supports_images`` 为 false
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            if not self.supports_images:
                raise ModelCapabilityError(
                    "当前配置未开启 supports_images；请在 config.toml 的 llm 段中为该 profile 设置 supports_images = true。"
                )

            self._last_reasoning_content = None

            # Format messages with image support
            formatted_messages = self.format_messages(messages, supports_images=True)

            # Ensure the last message is from the user to attach images
            if not formatted_messages or formatted_messages[-1]["role"] != "user":
                raise ValueError(
                    "The last message must be from the user to attach images"
                )

            # Process the last user message to include images
            last_message = formatted_messages[-1]

            # Convert content to multimodal format if needed
            content = last_message["content"]
            multimodal_content = (
                [{"type": "text", "text": content}]
                if isinstance(content, str)
                else content
                if isinstance(content, list)
                else []
            )

            # Add images to content
            for image in images:
                if isinstance(image, str):
                    multimodal_content.append(
                        {"type": "image_url", "image_url": {"url": image}}
                    )
                elif isinstance(image, dict) and "url" in image:
                    multimodal_content.append({"type": "image_url", "image_url": image})
                elif isinstance(image, dict) and "image_url" in image:
                    multimodal_content.append(image)
                else:
                    raise ValueError(f"Unsupported image format: {image}")

            # Update the message with multimodal content
            last_message["content"] = multimodal_content

            # Add system messages if provided
            if system_msgs:
                all_messages = (
                    self.format_messages(system_msgs, supports_images=True)
                    + formatted_messages
                )
            else:
                all_messages = formatted_messages

            # Calculate tokens and check limits
            input_tokens = self.count_message_tokens(all_messages)
            if not self.check_token_limit(input_tokens):
                raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))

            # Set up API parameters
            params = {
                "model": self.model,
                "messages": all_messages,
                "stream": stream,
            }

            # Add model-specific parameters
            if self.use_max_completion_tokens:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            # Handle non-streaming request
            if not stream:
                response = await self.client.chat.completions.create(**params)

                msg = response.choices[0].message if response.choices else None
                if not msg or not (msg.content or _extract_reasoning_content(msg)):
                    raise ValueError("Empty or invalid response from LLM")

                self.update_token_count(response.usage.prompt_tokens)
                self._last_reasoning_content = _extract_reasoning_content(msg)
                return msg.content or ""

            # Handle streaming request
            self.update_token_count(input_tokens)
            response = await self.client.chat.completions.create(**params)

            print("-"*20)
            collected_messages = []
            completion_text = ""
            reasoning_text = ""

            self._emit_event(
                "llm_stream_start",
                {
                    "model": self.model,
                    "input_tokens": input_tokens,
                    "stream": True,
                    "with_images": True,
                },
                on_event=on_event,
            )

            async for chunk in response:
                # Some OpenAI-compatible providers may yield chunks with empty `choices`
                # (or with no `delta.content`). Guard to avoid `IndexError`.
                if not getattr(chunk, "choices", None):
                    continue

                choice0 = chunk.choices[0]
                delta = getattr(choice0, "delta", None)
                if delta is not None:
                    rchunk = _extract_reasoning_content(delta)
                    if rchunk:
                        reasoning_text += rchunk

                chunk_message = (
                    (getattr(delta, "content", None) or "") if delta is not None else ""
                )

                if not chunk_message:
                    continue

                collected_messages.append(chunk_message)
                completion_text += chunk_message
                print(chunk_message, end="", flush=True)
                self._emit_event(
                    "llm_stream_chunk",
                    {
                        "text": chunk_message,
                        "accumulated_text_len": len(completion_text),
                    },
                    on_event=on_event,
                )
            if not collected_messages:
                print("No print content from LLM")
            print()  # Newline after streaming
            print("-"*20)
            full_response = "".join(collected_messages).strip()

            if not full_response:
                raise ValueError("Empty response from streaming LLM")

            self._last_reasoning_content = reasoning_text or None

            completion_tokens = self.count_tokens(completion_text)
            self._emit_event(
                "llm_stream_end",
                {
                    "model": self.model,
                    "estimated_completion_tokens": completion_tokens,
                    "output_text_len": len(full_response),
                    "with_images": True,
                },
                on_event=on_event,
            )
            return full_response

        except TokenLimitExceeded:
            raise
        except ModelCapabilityError:
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_with_images: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_with_images: {e}")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,  # type: ignore
        temperature: Optional[float] = None,
        verbose: bool = False,
        **kwargs,
    ) -> ChatCompletionMessage | None:
        """
        Ask LLM using functions/tools and return the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds
            tools: List of tools to use
            tool_choice: Tool choice strategy
            temperature: Sampling temperature for the response
            **kwargs: Additional completion arguments
            verbose (bool): Whether to print the messages.
        Returns:
            ChatCompletionMessage: The model's response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If tools, tool_choice, or messages are invalid
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # Validate tool_choice
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            self._last_reasoning_content = None

            # Check if the model supports images
            supports_images = self.supports_images

            # Format messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)
            
            if verbose:
                logger.info(f"[VERBOSE] caller={get_real_caller()} model={self.model} messages={messages}")
            
            # Calculate input token count
            input_tokens = self.count_message_tokens(messages)

            # If there are tools, calculate token count for tool descriptions
            tools_tokens = 0
            if tools:
                for tool in tools:
                    tools_tokens += self.count_tokens(str(tool))

            input_tokens += tools_tokens

            # Check if token limits are exceeded
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # Raise a special exception that won't be retried
                raise TokenLimitExceeded(error_message)

            # Validate tools if provided
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")

            # Set up the completion request
            params = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "timeout": timeout,
                **kwargs,
            }

            if self.use_max_completion_tokens:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            params["stream"] = False  # Always use non-streaming for tool requests
            response: ChatCompletion = await self.client.chat.completions.create(
                **params
            )

            # Check if response is valid
            if not response.choices or not response.choices[0].message:
                print("-"*20)
                print("No print content from LLM")
                print(response)
                print("-"*20)
                # raise ValueError("Invalid or empty response from LLM")
                return None
            # Update token counts
            self.update_token_count(
                response.usage.prompt_tokens, response.usage.completion_tokens
            )

            self._last_reasoning_content = _extract_reasoning_content(
                response.choices[0].message
            )

            return response.choices[0].message

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_tool: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_tool: {e}")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask_tool_stream(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,  # type: ignore
        temperature: Optional[float] = None,
        verbose: bool = False,
        on_event: Optional[EventCallback] = None,
        **kwargs,
    ) -> tuple[str, List[dict]]:
        """
        Ask LLM using tools with streaming and collect tool calls.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds
            tools: List of tools to use
            tool_choice: Tool choice strategy
            temperature: Sampling temperature for the response
            verbose (bool): Whether to print the messages.
            on_event: Optional event callback (sync/async). If not provided, uses `self.on_event`.
                Intended for streaming side-effects (front-end push, TTS, etc.). Failures are isolated.
            **kwargs: Additional completion arguments

        Returns:
            tuple[str, List[dict]]: (assistant text, collected tool calls)

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If tools, tool_choice, or streamed response is invalid/empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            self._last_reasoning_content = None

            supports_images = self.supports_images
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            if verbose:
                logger.info(
                    f"[VERBOSE] caller={get_real_caller()} model={self.model} messages={messages}"
                )

            input_tokens = self.count_message_tokens(messages)

            tools_tokens = 0
            if tools:
                for tool in tools:
                    tools_tokens += self.count_tokens(str(tool))
            input_tokens += tools_tokens

            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                raise TokenLimitExceeded(error_message)

            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")

            params = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "timeout": timeout,
                "stream": True,
                **kwargs,
            }

            if self.use_max_completion_tokens:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            self.update_token_count(input_tokens)
            response = await self.client.chat.completions.create(**params)

            print("-" * 20)
            collected_messages: List[str] = []
            completion_text = ""
            reasoning_text = ""
            tool_calls_buffer: dict[int, dict] = {}

            self._emit_event(
                "llm_stream_start",
                {
                    "model": self.model,
                    "input_tokens": input_tokens,
                    "stream": True,
                    "with_tools": True,
                    "tool_choice": tool_choice,
                },
                on_event=on_event,
            )

            async for chunk in response:
                if not getattr(chunk, "choices", None):
                    continue

                choice0 = chunk.choices[0]
                delta = getattr(choice0, "delta", None)
                if delta is None:
                    continue

                rchunk = _extract_reasoning_content(delta)
                if rchunk:
                    reasoning_text += rchunk

                chunk_message = getattr(delta, "content", None) or ""
                if chunk_message:
                    collected_messages.append(chunk_message)
                    completion_text += chunk_message
                    print(chunk_message, end="", flush=True)
                    self._emit_event(
                        "llm_stream_chunk",
                        {
                            "text": chunk_message,
                            "accumulated_text_len": len(completion_text),
                            "with_tools": True,
                        },
                        on_event=on_event,
                    )

                delta_tool_calls = getattr(delta, "tool_calls", None) or []
                for tc_delta in delta_tool_calls:
                    index = getattr(tc_delta, "index", None)
                    if index is None and isinstance(tc_delta, dict):
                        index = tc_delta.get("index")
                    if index is None:
                        index = 0

                    if index not in tool_calls_buffer:
                        tool_calls_buffer[index] = {
                            "id": "",
                            "name": "",
                            "arguments": "",
                        }

                    tc_id = getattr(tc_delta, "id", None)
                    if tc_id is None and isinstance(tc_delta, dict):
                        tc_id = tc_delta.get("id")
                    if tc_id:
                        tool_calls_buffer[index]["id"] = tc_id

                    func = getattr(tc_delta, "function", None)
                    if func is None and isinstance(tc_delta, dict):
                        func = tc_delta.get("function")
                    if not func:
                        continue

                    func_name = getattr(func, "name", None)
                    if func_name is None and isinstance(func, dict):
                        func_name = func.get("name")
                    if func_name:
                        tool_calls_buffer[index]["name"] = func_name

                    func_arguments = getattr(func, "arguments", None)
                    if func_arguments is None and isinstance(func, dict):
                        func_arguments = func.get("arguments")
                    if func_arguments:
                        tool_calls_buffer[index]["arguments"] += func_arguments

            print()  # Newline after streaming
            print("-" * 20)

            full_response = "".join(collected_messages).strip()

            tool_calls: List[dict] = []
            for idx in sorted(tool_calls_buffer.keys()):
                tc = tool_calls_buffer[idx]
                tool_id = tc.get("id") or f"call_{uuid.uuid4().hex[:24]}"
                tool_name = tc.get("name") or ""
                arguments_str = tc.get("arguments") or ""

                parsed_args = {}
                parse_error = None
                if arguments_str:
                    try:
                        parsed_args = json.loads(arguments_str)
                    except json.JSONDecodeError as exc:
                        parse_error = str(exc)
                        logger.error(
                            "Failed to parse streamed tool arguments. name=%s, args_preview=%s, error=%s",
                            tool_name,
                            arguments_str[:200],
                            parse_error,
                        )

                tool_call_payload = {
                    "id": tool_id,
                    "name": tool_name,
                    "arguments": parsed_args,
                }
                if parse_error:
                    tool_call_payload["_parse_error"] = parse_error
                    tool_call_payload["_raw_arguments"] = arguments_str

                tool_calls.append(tool_call_payload)

            if not full_response and not tool_calls:
                raise ValueError("Empty response from streaming tool LLM")

            self._last_reasoning_content = reasoning_text or None

            # Estimate completion tokens for streaming response (text + tool arguments).
            completion_text = full_response + "".join(
                (tc.get("name") or "") + (tc.get("_raw_arguments") or "")
                for tc in tool_calls
            )
            completion_tokens = self.count_tokens(completion_text)
            logger.info(
                "[LLM] %s estimated completion tokens for streaming tool response: %s",
                self.model,
                completion_tokens,
            )
            self.total_completion_tokens += completion_tokens

            self._emit_event(
                "llm_stream_end",
                {
                    "model": self.model,
                    "estimated_completion_tokens": completion_tokens,
                    "output_text_len": len(full_response),
                    "with_tools": True,
                },
                on_event=on_event,
            )
            return full_response, tool_calls

        except TokenLimitExceeded:
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_tool_stream: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_tool_stream: {e}")
            raise


__all__ = ["LLM"]
