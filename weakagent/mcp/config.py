"""Load MCP server definitions from config.toml."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from weakagent.config.settings import config


@dataclass
class MCPServerSpec:
    """One MCP server connection."""

    id: str
    transport: str = "streamable_http"  # streamable_http | sse | stdio
    url: Optional[str] = None
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    sync_key: Optional[str] = None
    enabled: bool = True

    def resolved_url(self) -> str:
        """Build URL with sync_key query param when configured."""
        if not self.url:
            raise ValueError(f"MCP server '{self.id}' has no url")
        if not self.sync_key:
            return self.url
        parsed = urlparse(self.url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query.setdefault("sync_key", self.sync_key)
        return urlunparse(parsed._replace(query=urlencode(query)))

    def resolved_headers(self) -> Dict[str, str]:
        """Merge custom headers with X-Sync-Key when sync_key is set."""
        out = dict(self.headers)
        if self.sync_key and "X-Sync-Key" not in out and "x-sync-key" not in out:
            out["X-Sync-Key"] = self.sync_key
        return out


@dataclass
class MCPSettings:
    enabled: bool = False
    servers: List[MCPServerSpec] = field(default_factory=list)


def _mcp_settings_from_section(section: Dict[str, Any]) -> MCPSettings:
    if not isinstance(section, dict):
        return MCPSettings()

    servers: List[MCPServerSpec] = []

    for item in section.get("servers") or []:
        if not isinstance(item, dict):
            continue
        servers.append(_spec_from_dict(item))

    if not servers and (section.get("server_url") or section.get("url")):
        servers.append(
            MCPServerSpec(
                id=str(section.get("server_id") or section.get("id") or "default"),
                transport=str(section.get("transport") or "streamable_http"),
                url=section.get("server_url") or section.get("url"),
                sync_key=section.get("sync_key"),
                headers=_as_str_dict(section.get("headers")),
                enabled=True,
            )
        )

    has_servers = bool(servers)
    enabled = bool(section.get("enabled", False)) or has_servers
    return MCPSettings(enabled=enabled, servers=[s for s in servers if s.enabled])


def load_mcp_settings(config_path: Optional[str] = None) -> MCPSettings:
    """Parse [mcp] from global config, or an alternate TOML path when given."""
    if config_path is None:
        return _mcp_settings_from_section(config.mcp_raw)

    path = config.resolve_path(config_path)
    if not path.exists():
        return MCPSettings()
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    section = raw.get("mcp") or {}
    return _mcp_settings_from_section(section)


def _spec_from_dict(data: Dict[str, Any]) -> MCPServerSpec:
    return MCPServerSpec(
        id=str(data.get("id") or data.get("server_id") or "mcp"),
        transport=str(data.get("transport") or "streamable_http"),
        url=data.get("url") or data.get("server_url"),
        command=data.get("command"),
        args=[str(a) for a in (data.get("args") or [])],
        headers=_as_str_dict(data.get("headers")),
        sync_key=data.get("sync_key"),
        enabled=bool(data.get("enabled", True)),
    )


def _as_str_dict(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items()}
