"""MCP client: connect external servers and expose tools to weakagent agents."""

from __future__ import annotations

import re
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import ListToolsResult, TextContent

from weakagent.mcp.config import MCPServerSpec, MCPSettings, load_mcp_settings
from weakagent.tools.base import BaseTool, ToolExecutionResult
from weakagent.tools.tool_collection import ToolCollection
from weakagent.utils.logger import get_logger

logger = get_logger(__name__)


def _make_mcp_client_tool(
    *,
    name: str,
    description: str,
    parameters: dict,
    session: ClientSession,
    server_id: str,
    original_name: str,
) -> "MCPClientTool":
    """Build a tool instance (BaseTool uses class-level metadata)."""

    class _BoundMCPClientTool(MCPClientTool):
        pass

    _BoundMCPClientTool.name = name
    _BoundMCPClientTool.description = description
    _BoundMCPClientTool.parameters = parameters

    inst = _BoundMCPClientTool()
    inst.session = session
    inst.server_id = server_id
    inst.original_name = original_name
    return inst


class MCPClientTool(BaseTool):
    """Proxy that calls a tool on a remote MCP server."""

    session: Optional[ClientSession] = None
    server_id: str = ""
    original_name: str = ""

    async def execute(self, **kwargs) -> ToolExecutionResult:
        if not self.session:
            return self.fail_response("Not connected to MCP server")
        try:
            logger.info("MCP tool %s/%s args=%s", self.server_id, self.original_name, kwargs)
            result = await self.session.call_tool(self.original_name, kwargs)
            parts = [
                item.text
                for item in result.content
                if isinstance(item, TextContent) and item.text
            ]
            return self.success_response("\n".join(parts) if parts else "No output returned.")
        except Exception as e:
            logger.exception("MCP tool call failed")
            return self.fail_response(f"MCP tool error: {e}")


class MCPClients(ToolCollection):
    """Connect to MCP servers and register remote tools locally."""

    sessions: Dict[str, ClientSession]
    exit_stacks: Dict[str, AsyncExitStack]

    def __init__(self, *tools: BaseTool):
        super().__init__(*tools)
        self.sessions = {}
        self.exit_stacks = {}

    async def connect_streamable_http(
        self,
        server_url: str,
        server_id: str = "",
        *,
        headers: Optional[Dict[str, str]] = None,
        terminate_on_close: bool = True,
    ) -> None:
        """Connect via Streamable HTTP (POST/GET /mcp, mcp-session-id)."""
        if not server_url:
            raise ValueError("server_url is required")
        server_id = server_id or server_url
        await self._connect(
            server_id,
            lambda stack: self._open_streamable_http(
                stack, server_url, headers=headers, terminate_on_close=terminate_on_close
            ),
        )

    async def connect_sse(
        self,
        server_url: str,
        server_id: str = "",
        *,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Connect via legacy MCP SSE transport."""
        if not server_url:
            raise ValueError("server_url is required")
        server_id = server_id or server_url
        await self._connect(
            server_id,
            lambda stack: stack.enter_async_context(sse_client(server_url, headers=headers)),
        )

    async def connect_stdio(
        self,
        command: str,
        args: Optional[List[str]] = None,
        server_id: str = "",
    ) -> None:
        """Connect via stdio subprocess."""
        if not command:
            raise ValueError("command is required")
        server_id = server_id or command
        params = StdioServerParameters(command=command, args=args or [])

        async def _open(stack: AsyncExitStack):
            transport = await stack.enter_async_context(stdio_client(params))
            return transport

        await self._connect(server_id, _open)

    async def connect_spec(self, spec: MCPServerSpec) -> None:
        """Connect using an MCPServerSpec from config."""
        transport = (spec.transport or "streamable_http").lower()
        if transport in ("streamable_http", "http", "streamable-http"):
            await self.connect_streamable_http(
                spec.resolved_url(),
                spec.id,
                headers=spec.resolved_headers(),
            )
        elif transport == "sse":
            await self.connect_sse(
                spec.resolved_url(),
                spec.id,
                headers=spec.resolved_headers(),
            )
        elif transport == "stdio":
            await self.connect_stdio(spec.command or "", spec.args, spec.id)
        else:
            raise ValueError(f"Unsupported MCP transport: {spec.transport}")

    async def connect_from_config(
        self,
        settings: Optional[MCPSettings] = None,
    ) -> List[str]:
        """Connect all servers from config.toml. Returns connected server ids."""
        settings = settings or load_mcp_settings()
        connected: List[str] = []
        for spec in settings.servers:
            try:
                await self.connect_spec(spec)
                connected.append(spec.id)
            except Exception as e:
                logger.error("Failed to connect MCP server %s: %s", spec.id, e)
        return connected

    async def _open_streamable_http(
        self,
        stack: AsyncExitStack,
        server_url: str,
        *,
        headers: Optional[Dict[str, str]],
        terminate_on_close: bool,
    ):
        http_client = await stack.enter_async_context(
            httpx.AsyncClient(headers=headers or {}, timeout=60.0)
        )
        ctx = streamable_http_client(
            server_url,
            http_client=http_client,
            terminate_on_close=terminate_on_close,
        )
        read_stream, write_stream, _get_session_id = await stack.enter_async_context(ctx)
        return read_stream, write_stream

    async def _connect(self, server_id: str, open_streams) -> None:
        if server_id in self.sessions:
            await self.disconnect(server_id)

        exit_stack = AsyncExitStack()
        self.exit_stacks[server_id] = exit_stack

        streams = await open_streams(exit_stack)
        session = await exit_stack.enter_async_context(ClientSession(*streams))
        self.sessions[server_id] = session
        await self._initialize_and_list_tools(server_id)

    async def _initialize_and_list_tools(self, server_id: str) -> None:
        session = self.sessions.get(server_id)
        if not session:
            raise RuntimeError(f"Session not initialized for {server_id}")

        await session.initialize()
        response = await session.list_tools()

        for tool in response.tools:
            original_name = tool.name
            tool_name = self._sanitize_tool_name(f"mcp_{server_id}_{original_name}")
            server_tool = _make_mcp_client_tool(
                name=tool_name,
                description=tool.description or f"MCP tool {original_name} on {server_id}",
                parameters=tool.inputSchema or {"type": "object", "properties": {}},
                session=session,
                server_id=server_id,
                original_name=original_name,
            )
            self.add_tool(server_tool)

        logger.info(
            "MCP server %s: registered tools %s",
            server_id,
            [t.name for t in response.tools],
        )

    @staticmethod
    def _sanitize_tool_name(name: str) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")
        return sanitized[:64] if len(sanitized) > 64 else sanitized

    async def list_tools(self) -> ListToolsResult:
        tools_result = ListToolsResult(tools=[])
        for session in self.sessions.values():
            response = await session.list_tools()
            tools_result.tools += response.tools
        return tools_result

    async def disconnect(self, server_id: str = "") -> None:
        if server_id:
            if server_id not in self.sessions:
                return
            try:
                exit_stack = self.exit_stacks.pop(server_id, None)
                if exit_stack:
                    await exit_stack.aclose()
            except Exception as e:
                logger.warning("MCP disconnect %s: %s", server_id, e)
            self.sessions.pop(server_id, None)
            self.tool_map = {
                k: v for k, v in self.tool_map.items() if getattr(v, "server_id", None) != server_id
            }
            self.tools = tuple(self.tool_map.values())
            logger.info("Disconnected MCP server %s", server_id)
            return

        for sid in sorted(list(self.sessions.keys())):
            await self.disconnect(sid)


async def create_mcp_clients_from_config() -> MCPClients:
    """Factory: load config and connect enabled MCP servers."""
    settings = load_mcp_settings()
    clients = MCPClients()
    if settings.enabled and settings.servers:
        await clients.connect_from_config(settings)
    return clients
