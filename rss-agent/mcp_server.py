"""
Standalone MCP server for the RSS AI Agent tools.

Exposes the five tools as proper MCP (Model Context Protocol) endpoints
so any MCP-compatible client (Claude Desktop, Cursor, etc.) can call them.

Usage:
    python mcp_server.py          # stdio transport (default for MCP clients)

Requires:  pip install mcp
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    import mcp.types as mcp_types

    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False

from rich.console import Console
from mcp_tools import check_rss_availability, web_search_rss_links
from database import RSSDatabase

console = Console()


def _ts() -> str:
    return f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim]"


TOOLS: list[dict] = [
    {
        "name": "search_rss_links",
        "description": (
            "Search the web for AI-related RSS feed URLs using DuckDuckGo. "
            "Returns a deduplicated list of discovered URLs."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query for RSS feeds with optionable ai_rule instructions"},
                "max_results": {"type": "integer", "default": 25},
            },
            "required": ["query"],
        },
    },
    {
        "name": "check_rss_availability",
        "description": (
            "Check whether an RSS/Atom feed URL is reachable and valid. "
            "Returns feed title, item count, and HTTP status."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The RSS/Atom feed URL to verify"},
                "timeout": {"type": "integer", "default": 12},
            },
            "required": ["url"],
        },
    },
    {
        "name": "get_rss_links",
        "description": (
            "Get current RSS links from the database. "
            "Returns a list of links with their status and metadata."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 50, "description": "Maximum number of links to return"},
            },
            "required": [],
        },
    },
    {
        "name": "delete_rss_link",
        "description": (
            "Delete an RSS link from the database by URL. "
            "Returns confirmation of deletion."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The RSS feed URL to delete"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "add_rss_link",
        "description": (
            "Add a new RSS link to the database. "
            "Returns confirmation if added or if it already exists."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The RSS feed URL to add"},
                "title": {"type": "string", "description": "Optional title for the feed"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional list of tags"},
            },
            "required": ["url"],
        },
    },
]


def build_server(db: RSSDatabase):
    if not _MCP_AVAILABLE:
        console.print("[bold red]❌ 'mcp' package not installed. Run: pip install mcp[/bold red]")
        sys.exit(1)

    server = Server("rss-ai-agent")

    @server.list_tools()
    async def list_tools():
        return [
            mcp_types.Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"])
            for t in TOOLS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        if name == "search_rss_links":
            query = arguments["query"]
            max_results = int(arguments.get("max_results", 25))
            result = web_search_rss_links(query, max_results=max_results)
            new_urls = [u for u in result.links_found if not db.url_exists(u)]
            text = (
                    f"Search: '{query}'\n"
                    f"Found: {len(result.links_found)} | New: {len(new_urls)}\n\n"
                    + "\n".join(f"  {u}" for u in result.links_found[:30])
            )
            return [mcp_types.TextContent(type="text", text=text)]

        if name == "check_rss_availability":
            url = arguments["url"]
            timeout = int(arguments.get("timeout", 12))
            result = check_rss_availability(url, timeout=timeout)
            if result.is_available:
                db.add_link(url=url, title=result.feed_title, tags=["ai"])
                db.update_availability(
                    url=url, is_available=True,
                    http_status=result.http_status,
                    feed_title=result.feed_title,
                    item_count=result.feed_item_count

                )
                status = f"✅ available | '{result.feed_title}' | {result.feed_item_count} items"
            else:
                status = f"❌ unavailable | {result.error}"
            text = f"URL: {url}\nStatus: {status}\nDB: {db.get_stats()}"
            return [mcp_types.TextContent(type="text", text=text)]

        if name == "get_rss_links":
            limit = int(arguments.get("limit", 50))
            links = db.get_all_links()[:limit]
            if not links:
                text = "No RSS links in database."
            else:
                lines = [f"Current RSS Links (showing {len(links)}):", ""]
                for lnk in links:
                    status = "✅" if lnk.is_available else "❌"
                    title = f" [{lnk.title}]" if lnk.title else ""
                    items = f" ({lnk.feed_item_count} items)" if lnk.feed_item_count is not None else ""
                    lines.append(f"  {status} {lnk.url}{title}{items}")
                text = "\n".join(lines)
            return [mcp_types.TextContent(type="text", text=text)]

        if name == "delete_rss_link":
            url = arguments["url"]
            deleted = db.delete_link(url)
            if deleted:
                stats = db.get_stats()
                text = f"✅ Deleted RSS link: {url}\nDB now has {stats['total']} total links."
            else:
                text = f"❌ Link not found in database: {url}"
            return [mcp_types.TextContent(type="text", text=text)]

        if name == "add_rss_link":
            url = arguments["url"]
            title = arguments.get("title", "")
            tags = arguments.get("tags", [])
            if db.url_exists(url):
                text = f"❌ Link already exists in database: {url}"
            else:
                db.add_link(url=url, title=title, tags=tags)
                text = f"✅ Added new RSS link: {url}\nDB now has {db.get_stats()['total']} total links."
            return [mcp_types.TextContent(type="text", text=text)]

        return [mcp_types.TextContent(type="text", text=f"Unknown tool: {name}")]

    return server


async def _run_stdio(db: RSSDatabase) -> None:
    server = build_server(db)
    opts = server.create_initialization_options()
    async with stdio_server() as (r, w):
        await server.run(r, w, opts, raise_exceptions=True)


def main() -> None:
    if not _MCP_AVAILABLE:
        console.print("[bold red]❌ Install the mcp package:  pip install mcp[/bold red]")
        sys.exit(1)
    parser = argparse.ArgumentParser(description="RSS AI Agent – MCP server (stdio)")
    parser.add_argument("--db", default=None)
    args = parser.parse_args()

    db = RSSDatabase(args.db) if args.db else RSSDatabase()
    console.print(f"{_ts()} [bold cyan]🚀 MCP server starting – {db.get_stats()['total']} links in DB[/bold cyan]")

    import asyncio
    asyncio.run(_run_stdio(db))


if __name__ == "__main__":
    main()
