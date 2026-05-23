"""MCP tools bridge — use weakagent.mcp for client; this re-exports for convenience."""

from weakagent.mcp import MCPClients, create_mcp_clients_from_config, load_mcp_settings

__all__ = ["MCPClients", "create_mcp_clients_from_config", "load_mcp_settings"]
