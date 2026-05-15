import threading
import tomllib
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


def _resolve_config_toml() -> Path:
    """定位 `config.toml`：优先当前工作目录，其次与 `weakagent` 包并列的仓库根（开发布局）。"""
    here = Path(__file__).resolve()
    cwd_cfg = Path.cwd() / "config.toml"
    if cwd_cfg.exists():
        return cwd_cfg
    # 开发：仓库根与 `weakagent/` 子目录并列时
    repo_root_cfg = here.parent.parent.parent / "config.toml"
    if repo_root_cfg.exists():
        return repo_root_cfg
    # 包内（若随包分发默认配置）
    pkg_cfg = here.parent / "config.toml"
    if pkg_cfg.exists():
        return pkg_cfg
    raise FileNotFoundError(
        "未找到 config.toml：请在运行目录放置该文件，或于项目根目录保留 config.toml。"
    )


def get_project_root() -> Path:
    """含 `config.toml` 的目录。"""
    return _resolve_config_toml().parent


PROJECT_ROOT = get_project_root()


class LLMSettings(BaseModel):
    """OpenAI SDK（官方或 OpenAI 兼容 HTTP 端点）。"""

    model_config = ConfigDict(extra="ignore")

    model: str = Field(..., description="Model name")
    base_url: str = Field(
        ...,
        description="OpenAI API base URL（官方 https://api.openai.com/v1 或兼容服务）",
    )
    api_key: str = Field(..., description="API key")
    max_tokens: int = Field(4096, description="Maximum number of tokens per request")
    max_input_tokens: Optional[int] = Field(
        None,
        description="Maximum input tokens to use across all requests (None for unlimited)",
    )
    temperature: float = Field(1.0, description="Sampling temperature")
    supports_images: bool = Field(
        False, description="模型是否支持图像输入（多模态）；需与端点实际能力一致"
    )
    use_max_completion_tokens: bool = Field(
        False,
        description="是否使用 max_completion_tokens（如 o1 等推理型接口），否则用 max_tokens",
    )

    # Optional: token window management (prompt + completion must fit context_window)
    context_window: Optional[int] = Field(
        None,
        description="模型上下文窗口大小（token）。若为 None 则不做窗口检测/压缩。",
    )
    reserve_completion_tokens: Optional[int] = Field(
        None,
        description="为输出预留的 completion tokens。默认回退到 max_tokens。",
    )


class SearchSettings(BaseModel):
    """可选的网页搜索配置（`config.toml` 中 `[search]` 段）。"""

    model_config = ConfigDict(extra="ignore")

    engine: str = Field(
        default="duckduckgo",
        description="首选搜索引擎：google | baidu | duckduckgo | bing",
    )
    fallback_engines: List[str] = Field(
        default_factory=lambda: ["bing", "google", "baidu"],
        description="主引擎失败时依次尝试的后备引擎",
    )
    lang: str = Field(default="en", description="语言代码")
    country: str = Field(default="us", description="国家/地区代码")
    retry_delay: int = Field(default=60, ge=0, description="整轮引擎失败后重试间隔（秒）")
    max_retries: int = Field(default=3, ge=0, description="整轮重试次数上限")


class _AppConfig(BaseModel):
    """应用级配置（LLM + 可选搜索等）。"""

    llm: Dict[str, LLMSettings]
    search: SearchSettings = Field(default_factory=SearchSettings)


class Config:
    _instance: Optional["Config"] = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._config: Optional[_AppConfig] = None
                    self._load_initial_config()
                    self._initialized = True

    @staticmethod
    def _get_config_path() -> Path:
        return _resolve_config_toml()

    def _load_config(self) -> dict:
        config_path = self._get_config_path()
        with config_path.open("rb") as f:
            return tomllib.load(f)

    def _load_initial_config(self) -> None:
        raw_config = self._load_config()
        base_llm = raw_config.get("llm", {})
        llm_overrides = {
            k: v for k, v in base_llm.items() if isinstance(v, dict)
        }

        default_settings = {
            "model": base_llm.get("model"),
            "base_url": base_llm.get("base_url"),
            "api_key": base_llm.get("api_key"),
            "max_tokens": base_llm.get("max_tokens", 4096),
            "max_input_tokens": base_llm.get("max_input_tokens"),
            "temperature": base_llm.get("temperature", 1.0),
            "supports_images": base_llm.get("supports_images", False),
            "use_max_completion_tokens": base_llm.get("use_max_completion_tokens", False),
            "context_window": base_llm.get("context_window"),
            "reserve_completion_tokens": base_llm.get("reserve_completion_tokens"),
        }

        llm_dict: Dict[str, LLMSettings] = {
            "default": LLMSettings(**default_settings),
            **{
                name: LLMSettings(**{**default_settings, **override_config})
                for name, override_config in llm_overrides.items()
            },
        }

        search_raw = raw_config.get("search") or {}
        if not isinstance(search_raw, dict):
            search_raw = {}
        search_settings = SearchSettings(**search_raw)

        self._config = _AppConfig(llm=llm_dict, search=search_settings)

    @property
    def llm(self) -> Dict[str, LLMSettings]:
        assert self._config is not None
        return self._config.llm

    @property
    def search_config(self) -> SearchSettings:
        assert self._config is not None
        return self._config.search


config = Config()
