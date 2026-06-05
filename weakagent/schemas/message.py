"""兼容层：统一使用 ``weakagent.memory.message`` 中的 Message。"""

from weakagent.memory.message import Message, ROLE_TYPE, ROLE_VALUES, Role

__all__ = ["Message", "Role", "ROLE_TYPE", "ROLE_VALUES"]
