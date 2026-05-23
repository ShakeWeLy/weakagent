"""
Connect external MCP servers from config.toml and list / call tools.

Record App example (Streamable HTTP):
  - Server: http://localhost:3001/mcp
  - Auth: sync_key in URL or X-Sync-Key header

Run (Record App server must be up):

    python examples/9_mcp_demo.py
    python examples/9_mcp_demo.py --call list_todos
"""

from __future__ import annotations

import argparse
import asyncio

from weakagent.mcp import MCPClients, create_mcp_clients_from_config, load_mcp_settings


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--call",
        help="Call one MCP tool by original name (e.g. list_todos)",
    )
    args = parser.parse_args()

    settings = load_mcp_settings()
    print("MCP enabled:", settings.enabled)
    print("Servers:", [(s.id, s.transport, s.resolved_url()) for s in settings.servers])

    clients = await create_mcp_clients_from_config()
    try:
        if not clients.tool_map:
            print("No MCP tools registered. Check [mcp] in config.toml and that the server is running.")
            return

        print("\nRegistered tools:")
        for name, tool in clients.tool_map.items():
            orig = getattr(tool, "original_name", "?")
            sid = getattr(tool, "server_id", "?")
            print(f"  {name}  (server={sid}, mcp={orig})")

        if args.call:
            target = None
            for name, tool in clients.tool_map.items():
                if getattr(tool, "original_name", None) == args.call:
                    target = tool
                    break
            if not target:
                print(f"Tool not found: {args.call}")
                return
            result = await target.execute()
            print("\nResult:", result.output or result.error)
    finally:
        await clients.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
