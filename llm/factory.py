"""``LLM`` 实例工厂：非单例，每次 ``create`` 得到独立 client。"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from config.settings import LLMSettings

if TYPE_CHECKING:
    from .llm import LLM


class LLMFactory:
    @staticmethod
    def create(
        config_name: str = "default",
        llm_config: Optional[LLMSettings] = None,
    ) -> LLM:
        from .llm import LLM

        return LLM(config_name=config_name, llm_config=llm_config)
