import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from weakagent.agent.chat import ChatAgent
from weakagent.llm.llm import LLM
from weakagent.utils.verbose import verbose_result


async def main():
    agent = ChatAgent(
        max_steps=1,
    )
    agent.llm = LLM(config_name="default")
    result = await agent.run("Who is the president of the United States?")
    verbose_result(result, agent)

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
