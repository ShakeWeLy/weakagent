from pydantic import BaseModel, Field
from .base import BaseTool, SearchArgs, ToolExecutionResult


class WebSearchArgs(BaseModel):
    query: str = Field(..., description="The query to search the web for")
    top_k: int = Field(default=5, ge=1, le=10, description="The number of results to return")


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web for information"
    args_model = WebSearchArgs

    async def execute(self, args: WebSearchArgs) -> ToolExecutionResult:
        try:
            result = {
                "query": args.query,
                "top_k": args.top_k,
            }
            return self.success_response(result)
        except Exception as e:
            return self.fail_response(str(e))
