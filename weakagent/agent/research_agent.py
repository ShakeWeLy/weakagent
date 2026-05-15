from typing import List, Optional, Union
from pydantic import Field

from weakagent.agent.brief_react import BriefReActAgent
from weakagent.tools import ToolCollection, WebSearch, Terminate, Summary
from weakagent.schemas.tool import TOOL_CHOICE_TYPE, ToolChoice
from weakagent.prompt.search import THINK_SYSTEM_PROMPT, ACT_SYSTEM_PROMPT


class ResearchAgent(BriefReActAgent):
    name: str = "research"
    description: str = "A research agent that can search the web for information."

    think_system_prompt: str = THINK_SYSTEM_PROMPT
    # think_next_step_prompt: str = THINK_NEXT_STEP_PROMPT  # use default
    act_system_prompt: str = ACT_SYSTEM_PROMPT
    # act_next_step_prompt: str = ACT_NEXT_STEP_PROMPT  # use default

    available_tools: ToolCollection = ToolCollection(
        WebSearch(), Summary(), Terminate())
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    max_steps: int = 30
    max_observe: Optional[Union[int, bool]] = None