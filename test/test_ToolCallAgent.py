import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import asyncio
import json
from types import SimpleNamespace

from weakagent.agent.toolcall import ToolCallAgent
from weakagent.schemas.tool import Function, ToolCall
from weakagent.llm.llm import LLM
from weakagent.tools.base import BaseTool, ToolExecutionResult
from weakagent.tools.terminate import Terminate
from weakagent.tools.tool_collection import ToolCollection
from weakagent.utils.verbose import verbose_result

class WeatherTool(BaseTool):
    name: str = "weather"
    description: str = "Get the weather for a given location"
    parameters: dict = {
        "location": {
            "type": "string",
            "description": "The location to get the weather for"
        }
    }
    async def execute(self, location: str) -> ToolExecutionResult:
        return ToolExecutionResult.ok(output=f"The weather in {location} is sunny")



async def main():
    agent = ToolCallAgent(
        max_steps=5,
        available_tools=ToolCollection(WeatherTool(), Terminate()),
    )
    # agent.llm = FakeToolLLM(build_fake_responses())
    agent.llm = LLM(config_name="default")

    result = await agent.run("Get the information for Tokyo")

    verbose_result(result, agent)
    final_tool_messages = [message for message in agent.messages if message.role == "tool"]
    assert len(final_tool_messages) == 2, "Expected two tool messages in memory"
    assert "The weather in Tokyo is sunny" in result
    assert "status: success" in result

    print("\nSmoke test completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())