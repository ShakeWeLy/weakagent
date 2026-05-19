from weakagent.tools.base import BaseTool
from weakagent.tools.base import ToolExecutionResult


# This tool is used to research the web for information by creating a research agent.
class AgentResearchTool(BaseTool):
    name: str = "agent_research"
    description: str = "A tool that can research the web for information"
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query to research the web for information."
            }
        },
        "required": ["query"],
    }

    async def execute(self, query: str) -> ToolExecutionResult:
        from weakagent.agent import ResearchAgent
        research_agent = ResearchAgent(
            name="research_agent",
            max_steps=10,
            max_observe=10,
            only_last_result=True,
            summarize_short_memory=True,
        )
        return ToolExecutionResult.ok(output=research_agent.run(query))