from __future__ import annotations

import json
from typing import Any, Dict, Optional

from weakagent.tools import ToolCollection
from weakagent.tools.scheduler import (
    CreateTaskTool,
    DeleteTaskTool,
    GetTaskTool,
    ListTasksTool,
    UpdateTaskTool,
)
from weakagent.tools import Terminate, AskHumanTool

from weakagent.utils.logger import get_logger
from weakagent.agent.brief_react import BriefReActAgent

logger = get_logger(__name__)


class TaskCrudAgent(BriefReActAgent):
    """Agent that performs task CRUD via scheduler tools.
    """

    name: str = "task_crud"
    description: str = "Structured task CRUD handler for scheduler request queue."

    available_tools: ToolCollection = ToolCollection(
        Terminate(),
        AskHumanTool(),
        CreateTaskTool(),
        GetTaskTool(),
        UpdateTaskTool(),
        ListTasksTool(),
        DeleteTaskTool(),
    )
    max_steps: int = 10
    
    async def act(self) -> str:
        return await super().act()