"""Attach MCP tools from config.toml to an agent ToolCollection."""

from __future__ import annotations

from typing import Optional, Tuple

from weakagent.mcp.client import MCPClients, create_mcp_clients_from_config
from weakagent.tools.tool_collection import ToolCollection
from weakagent.utils.logger import get_logger

logger = get_logger(__name__)


async def attach_mcp_to_tools(
    tools: ToolCollection,
) -> Tuple[ToolCollection, Optional[MCPClients]]:
    """
    Connect MCP servers from config and merge remote tools into *tools*.

    Returns:
        (tools, mcp_clients) — keep *mcp_clients* alive for the agent run, then
        ``await mcp_clients.disconnect()`` when done.
    """
    clients = await create_mcp_clients_from_config()
    if not clients.tool_map:
        return tools, None

    for tool in clients.tools:
        tools.add_tool(tool)
    logger.info("Attached %s MCP tools to agent", len(clients.tools))
    return tools, clients
