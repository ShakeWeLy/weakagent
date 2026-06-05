import threading
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

_CONFIG_WARNED = False

EnableThink = Literal["default", "enabled", "disabled"]


def _find_config_toml() -> Optional[Path]:
    """定位 `config.toml`；不存在则返回 None。"""
    here = Path(__file__).resolve()
    for candidate in (
        Path.cwd() / "config.toml",
        here.parent.parent.parent / "config.toml",
        here.parent / "config.toml",
    ):
        if candidate.exists():
            return candidate
    return None


def _default_project_root() -> Path:
    return Path.cwd()


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
    enable_think: EnableThink = Field(
        "default",
        description=(
            '思考/推理模式开关（兼容 DeepSeek 等：经 extra_body 发送 '
            '{"thinking":{"type":"enabled"|"disabled"}}）。'
            '"default" - do not send extra_body.thinking (provider default, e.g. DeepSeek’s default).'
            '"enabled" - sends extra_body={"thinking": {"type": "enabled"}}.'
            '"disabled" - sends extra_body={"thinking": {"type": "disabled"}}.'
        ),
    )
    context_window: Optional[int] = Field(
        None,
        description="模型上下文窗口大小（token）。若为 None 则不做窗口检测/压缩。",
    )
    reserve_completion_tokens: Optional[int] = Field(
        None,
        description="为输出预留的 completion tokens。默认回退到 max_tokens。",
    )

    @field_validator("enable_think", mode="before")
    @classmethod
    def _coerce_enable_think(cls, v: object) -> Union[EnableThink, object]:
        if v is True:
            return "enabled"
        if v is False:
            return "disabled"
        if v is None:
            return "default"
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("default", "enabled", "disabled"):
                return s  # type: ignore[return-value]
        return v


class SearchSettings(BaseModel):
    """网页搜索配置（`config.toml` 中 `[search]` 段）。"""

    model_config = ConfigDict(extra="ignore")

    engine: str = Field(default="duckduckgo")
    fallback_engines: List[str] = Field(
        default_factory=lambda: ["bing", "google", "baidu"],
    )
    lang: str = Field(default="en")
    country: str = Field(default="us")
    retry_delay: int = Field(default=60, ge=0)
    max_retries: int = Field(default=3, ge=0)


class DbPathSettings(BaseModel):
    """各 SQLite 段（`[scheduler]`、`[conversation]` 等）。"""

    model_config = ConfigDict(extra="ignore")

    db_path: str = "weakagent.sqlite3"


class SkillsSettings(BaseModel):
    """`[skills]` 段。"""

    model_config = ConfigDict(extra="ignore")

    builtin_dir: Optional[str] = None
    custom_dir: Optional[str] = None


class _AppConfig(BaseModel):
    llm: Dict[str, LLMSettings] = Field(default_factory=dict)
    search: SearchSettings = Field(default_factory=SearchSettings)
    scheduler: DbPathSettings = Field(
        default_factory=lambda: DbPathSettings(db_path="tasks.sqlite3")
    )
    conversation: DbPathSettings = Field(default_factory=DbPathSettings)
    runtime_session: DbPathSettings = Field(default_factory=DbPathSettings)
    memory: DbPathSettings = Field(default_factory=DbPathSettings)
    skills: SkillsSettings = Field(default_factory=SkillsSettings)
    mcp: Dict[str, Any] = Field(default_factory=dict)


class Config:
    """全局配置单例；`config.toml` 可选，缺失时使用默认值并 warning。"""

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
                    self._config_path: Optional[Path] = None
                    self._raw: Dict[str, Any] = {}
                    self._config: Optional[_AppConfig] = None
                    self._load_initial_config()
                    self._initialized = True

    @property
    def config_path(self) -> Optional[Path]:
        return self._config_path

    @property
    def project_root(self) -> Path:
        if self._config_path is not None:
            return self._config_path.parent
        return _default_project_root()

    @property
    def raw(self) -> Dict[str, Any]:
        return self._raw

    def _warn_missing_config(self) -> None:
        global _CONFIG_WARNED
        if _CONFIG_WARNED:
            return
        _CONFIG_WARNED = True
        warnings.warn(
            "未找到 config.toml，使用内置默认配置；"
            "可在运行目录或项目根放置 config.toml 覆盖。",
            UserWarning,
            stacklevel=3,
        )

    def _load_raw(self) -> Dict[str, Any]:
        self._config_path = _find_config_toml()
        if self._config_path is None:
            self._warn_missing_config()
            return {}
        try:
            with self._config_path.open("rb") as f:
                return tomllib.load(f)
        except Exception as exc:
            warnings.warn(
                f"读取 config.toml 失败 ({self._config_path}): {exc}，使用默认配置。",
                UserWarning,
                stacklevel=3,
            )
            return {}

    @staticmethod
    def _parse_section(raw: Dict[str, Any], key: str, model: type[BaseModel], default: BaseModel) -> BaseModel:
        section = raw.get(key) or {}
        if not isinstance(section, dict):
            return default
        try:
            return model(**section)
        except Exception:
            return default

    @staticmethod
    def _build_llm_dict(raw: Dict[str, Any]) -> Dict[str, LLMSettings]:
        base_llm = raw.get("llm", {})
        if not isinstance(base_llm, dict):
            return {}

        llm_overrides = {k: v for k, v in base_llm.items() if isinstance(v, dict)}
        default_settings = {
            "model": base_llm.get("model"),
            "base_url": base_llm.get("base_url"),
            "api_key": base_llm.get("api_key"),
            "max_tokens": base_llm.get("max_tokens", 4096),
            "max_input_tokens": base_llm.get("max_input_tokens"),
            "temperature": base_llm.get("temperature", 1.0),
            "supports_images": base_llm.get("supports_images", False),
            "use_max_completion_tokens": base_llm.get("use_max_completion_tokens", False),
            "enable_think": base_llm.get("enable_think", "default"),
            "context_window": base_llm.get("context_window"),
            "reserve_completion_tokens": base_llm.get("reserve_completion_tokens"),
        }

        required = ("model", "base_url", "api_key")
        if not all(default_settings.get(k) for k in required):
            if llm_overrides:
                warnings.warn(
                    "[llm] 缺少 model/base_url/api_key，仅加载命名 profile。",
                    UserWarning,
                    stacklevel=3,
                )
            result: Dict[str, LLMSettings] = {}
            for name, override in llm_overrides.items():
                try:
                    result[name] = LLMSettings(**override)
                except Exception:
                    pass
            return result

        result = {
            "default": LLMSettings(**default_settings),
            **{
                name: LLMSettings(**{**default_settings, **override})
                for name, override in llm_overrides.items()
            },
        }
        return result

    def _load_initial_config(self) -> None:
        self._raw = self._load_raw()
        mcp_section = self._raw.get("mcp") or {}
        self._config = _AppConfig(
            llm=self._build_llm_dict(self._raw),
            search=self._parse_section(
                self._raw, "search", SearchSettings, SearchSettings()
            ),
            scheduler=self._parse_section(
                self._raw,
                "scheduler",
                DbPathSettings,
                DbPathSettings(db_path="tasks.sqlite3"),
            ),
            conversation=self._parse_section(
                self._raw, "conversation", DbPathSettings, DbPathSettings()
            ),
            runtime_session=self._parse_section(
                self._raw, "runtime_session", DbPathSettings, DbPathSettings()
            ),
            memory=self._parse_section(
                self._raw, "memory", DbPathSettings, DbPathSettings()
            ),
            skills=self._parse_section(
                self._raw, "skills", SkillsSettings, SkillsSettings()
            ),
            mcp=mcp_section if isinstance(mcp_section, dict) else {},
        )

    def resolve_path(self, path: Union[str, Path]) -> Path:
        """将相对路径解析到 project_root。"""
        p = Path(path)
        return p if p.is_absolute() else (self.project_root / p)

    def resolve_db_path(
        self,
        db_path: Union[str, Path],
        *,
        sections: tuple[str, ...] = (),
    ) -> Path:
        """绝对路径原样返回；否则按 sections 顺序查配置，最后相对 project_root。"""
        p = Path(db_path)
        if p.is_absolute():
            return p
        for section in sections:
            configured = self.get_section_db_path(section)
            if configured:
                return self.resolve_path(configured)
        return self.project_root / p

    def get_section_db_path(self, section: str) -> Optional[str]:
        assert self._config is not None
        mapping: Dict[str, DbPathSettings] = {
            "scheduler": self._config.scheduler,
            "conversation": self._config.conversation,
            "runtime_session": self._config.runtime_session,
            "session": self._config.runtime_session,
            "memory": self._config.memory,
        }
        settings = mapping.get(section)
        if settings is None:
            return None
        return settings.db_path

    @property
    def llm(self) -> Dict[str, LLMSettings]:
        assert self._config is not None
        return self._config.llm

    @property
    def search_config(self) -> SearchSettings:
        assert self._config is not None
        return self._config.search

    @property
    def scheduler(self) -> DbPathSettings:
        assert self._config is not None
        return self._config.scheduler

    @property
    def conversation(self) -> DbPathSettings:
        assert self._config is not None
        return self._config.conversation

    @property
    def runtime_session(self) -> DbPathSettings:
        assert self._config is not None
        return self._config.runtime_session

    @property
    def memory(self) -> DbPathSettings:
        assert self._config is not None
        return self._config.memory

    @property
    def skills(self) -> SkillsSettings:
        assert self._config is not None
        return self._config.skills

    @property
    def mcp_raw(self) -> Dict[str, Any]:
        assert self._config is not None
        return self._config.mcp


config = Config()


def get_project_root() -> Path:
    return config.project_root


PROJECT_ROOT = config.project_root
