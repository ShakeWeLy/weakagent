"""
Skills integration demo.

Shows the flow:
  skills/ + workspace/skills/ -> SkillLoader -> SkillManager
  -> system prompt <available_skills> -> LLM -> read(SKILL.md)

Run from project root:

    python examples/8_skills_demo.py
"""

from __future__ import annotations

import asyncio

from weakagent.agent import ToolCallAgent
from weakagent.llm.llm import LLM
from weakagent.schemas.message import Message
from weakagent.skills.manager import SkillManager
from weakagent.skills.service import SkillService
from weakagent.tools import ToolCollection, Terminate
from weakagent.tools.files import ReadFileTool


class SkillsDemoAgent(ToolCallAgent):
    name: str = "skills_demo"
    description: str = "Demo agent with skills catalog and read tool"
    system_prompt: str = (
        "You are a helpful assistant. When a task matches a skill, "
        "read its SKILL.md with the read tool and follow it."
    )
    available_tools: ToolCollection = ToolCollection(ReadFileTool(), Terminate())
    max_steps: int = 5
    skills_enabled: bool = True


async def inspect_catalog() -> None:
    mgr = SkillManager()
    svc = SkillService(mgr)

    print("=== Skill catalog ===")
    for row in svc.query():
        flag = "on" if row.get("enabled") else "off"
        print(f"  [{flag}] {row['name']}: {row.get('description', '')[:60]}")

    print("\n=== Prompt snippet ===")
    print(mgr.build_skills_prompt()[:800])
    if len(mgr.build_skills_prompt()) > 800:
        print("...(truncated)")


async def run_agent_turn() -> None:
    agent = SkillsDemoAgent(llm=LLM(config_name="fast"))
    effective = agent.with_skills_prompt(agent.system_prompt) or ""
    print("\n=== Effective system prompt (tail) ===")
    print(effective[-600:] if len(effective) > 600 else effective)

    request = "请用 demo-skill 的方式给我打个结构化招呼"
    agent.update_memory("user", request)
    print(f"\n=== User: {request} ===\n")

    result = await agent.run(request)
    print(f"=== Agent result ===\n{result}\n")


async def main() -> None:
    await inspect_catalog()
    await run_agent_turn()


if __name__ == "__main__":
    asyncio.run(main())
