
import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only needed for type checking; avoid runtime import cycles.
    from weakagent.agent.base import BaseAgent

def verbose_result(result: str, agent: "BaseAgent"):
    print("\n=== Run Result ===")
    print(result)

    print("\n=== Memory Trace ===")
    for idx, message in enumerate(agent.messages, start=1):
        tool_meta = f" ({message.name})" if message.name else ""
        print(f"{idx}. {message.role}{tool_meta}: {message.content}")


def get_real_caller():
    for frame in inspect.stack()[1:]:
        name = frame.function
        if name != "__call__":
            return name
    return "Unknown"
