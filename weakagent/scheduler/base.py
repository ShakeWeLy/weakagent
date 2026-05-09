from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any, TypeVar
from enum import Enum

T = TypeVar("T", bound=BaseModel)

class TaskType(str, Enum):
    CHAT = "chat"
    SQL = "sql"
    RETRIEVAL = "retrieval"
    REPORT = "report"
    TOOL_CALL = "tool_call"
    WORKFLOW = "workflow"
    OTHER = "other"

class TaskStatus(str, Enum):
    """The status of the task"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(int, Enum):
    """The priority of the task"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class TaskResult(BaseModel):
    success: bool
    output: Any = None
    error: Optional[str] = None

class BaseTask(BaseModel, ABC):
    """Base class for all tasks"""
    id: int = Field(..., description="The id of the task")
    user_id: int = Field(..., description="The id of the user")
    task_name: str = Field(..., description="The name of the task")
    task_type: TaskType = Field(..., description="The type of the task")
    description: Optional[str] = Field(None, description="The description of the task")
    max_retries: int = 3
    retry_count: int = 0
    status: TaskStatus = Field(..., description="The status of the task")
    timeout: Optional[int] = None
    depends_on: list[int] = []
    next_run_at: Optional[datetime]
    priority: TaskPriority = Field(..., description="The priority of the task")
    task_params: T = Field(..., description="The parameters of the task")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    prompt: Optional[str] = Field(None, description="The prompt of the task")

    @abstractmethod
    async def execute(self) -> TaskResult:
        """Execute the task"""
        raise NotImplementedError

    def to_dict(self) -> dict:
        """Convert the task to a dictionary"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict):
        return cls.model_validate(data)
