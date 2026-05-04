import os
from typing import Optional
from weakagent.config.settings import config
from weakagent.llm import LLM
from weakagent.schemas.message import Message

from weakagent.tools.base import BaseTool, ToolExecutionResult
from weakagent.utils.image import image_to_base64


class ReadImagesTool(BaseTool):
    name: str = "read_images"
    description: str = "Use when you need to read an image and return the content of the image, especially when you need to analyze the image"
    parameters: dict = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "The path to the image to read"
            },
            "prompt": {
                "type": "string",
                "description": "The prompt to use for analyzing the image, such as 'describe the image'"
            },
        },
        "required": ["image_path"],
    }
    async def execute(self, image_path: str, prompt: Optional[str] = None) -> ToolExecutionResult:
        if not config.llm["default"].supports_images:
            return self.fail_response("Model does not support images")

        if not os.path.exists(image_path):
            return self.fail_response(f"Image not found: {image_path}")
        if prompt is None:
            prompt = "describe the image"
        llm = LLM(config_name="default")
        image_url = image_to_base64(image_path)
        messages = [
            Message.system_message(prompt),
            Message.user_message(
                prompt,
                base64_image=image_url,
            ),
        ]
        answer = await llm.ask(messages=messages)
        return ToolExecutionResult(success=True, message="Images read successfully", data={"answer": answer})