"""
RSS AI Agent – PydanticAI agent with five MCP tools.

MCP Tools registered on the agent:
  • search_rss_links      – web search for AI RSS feeds
  • check_feed_availability – HTTP + feedparser availability check
  • get_rss_links         – list current RSS links from DB
  • delete_rss_link       – remove an RSS link from DB
  • add_rss_link          – add a new RSS link to DB

The agent is backed by Ollama via the OpenAI-compatible /v1 endpoint.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from rich.console import Console
from rich.rule import Rule

import config
from database import RSSDatabase
from mcp_tools import check_rss_availability, web_search_rss_links
from models import AgentRunResult

console = Console()

logger = logging.getLogger("rss_agent.agent")


def _ts() -> str:
    return f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim]"


# ── Agent dependencies ────────────────────────────────────────────────────────

@dataclass
class AgentDeps:
    """Injected into every tool call via RunContext."""
    db: RSSDatabase


# ── Model (Ollama) ────────────────────────────────────────────────────────────

_model = OpenAIChatModel(
    config.OLLAMA_MODEL,
    provider=OpenAIProvider(
        base_url=f"{config.OLLAMA_HOST}",
        api_key='ollama'
    )
)

# ── Agent ─────────────────────────────────────────────────────────────────────

rss_agent = Agent(
    _model,
    deps_type=AgentDeps,
    system_prompt=(
        "You are an AI RSS feed maintenance agent. Your tasks:\n"
        "1. Check all current links  returned by get_rss_links()."
        "3. Validate each feed using check_feed_availability(feed) in db is reachable and contains valid RSS/Atom feed content.\n"
        "5. Discover RSS feeds related to artificial intelligence, machine learning, "
        "   large language models, vision, object-detection and research big language/experimental models using "
        "search_rss_links(query).\n"
        "6. Report how many new feeds were added, how many removed and the "
        "   current database content and totals.\n"
        "Use optionable scoped instruction ai_rule in query.\n"
        "To add new link use with add_rss_link, to delete link use delete_rss_link."
    ),
    retries=5,
)

# Set model settings for more verbose/deterministic output if needed, but here we just ensure retries are set
rss_agent.model_settings = {"temperature": config.TEMPERATURE if hasattr(config, 'TEMPERATURE') else 0.55}

logging.info(
    "Initialized RSS Agent with model: %s, temperature: %.2f",
    config.OLLAMA_MODEL,
    rss_agent.model_settings.get("temperature", -1.1)
)


# ── MCP Tool 1: search ────────────────────────────────────────────────────────

@rss_agent.tool
async def search_rss_links(ctx: RunContext[AgentDeps], query: str) -> str:
    """
    MCP Tool – Search the web for AI-related RSS feed URLs.

    Args:
        query: Natural-language search query, e.g.
               "AI research blog RSS feed 2024"

    Returns:
        A text summary listing the discovered URLs split into
        'new' (not yet in DB) and 'already known' groups.
    """
    logger.info(f"Step: Searching for RSS links with query: '{query}'")
    result = web_search_rss_links(query)
    logger.info(f"Tool search_rss_links returned {len(result.links_found)} raw links")

    new_urls: List[str] = []
    known_urls: List[str] = []
    for url in result.links_found:
        (known_urls if ctx.deps.db.url_exists(url) else new_urls).append(url)

    result.new_links = new_urls
    result.duplicate_links = known_urls

    console.print(
        f"{_ts()} [green]🔍 search_rss_links:[/green] "
        f"{len(new_urls)} new | {len(known_urls)} already known"
    )

    lines = [
        f"Search query: '{query}'",
        f"Total URLs found : {len(result.links_found)}",
        f"New (not in DB)  : {len(new_urls)}",
        f"Already known    : {len(known_urls)}",
        "",
        "New URLs:",
    ]
    lines += [f"  - {u}" for u in new_urls[:25]]
    if known_urls:
        lines += ["", "Already-known URLs (skipping):"]
        lines += [f"  - {u}" for u in known_urls[:10]]

    return "\n".join(lines)


# ── MCP Tool 2: availability check ───────────────────────────────────────────

@rss_agent.tool
async def check_feed_availability(ctx: RunContext[AgentDeps], url: str) -> str:
    """
    MCP Tool – Check whether an RSS feed URL is reachable and valid,
    then persist the result in the database.

    Args:
        url: The RSS feed URL to verify.

    Returns:
        A human-readable status string with feed metadata and DB totals.
    """
    logger.info(f"Step: Checking availability for URL: {url}")
    result = check_rss_availability(url)

    if result.is_available:
        is_new, _ = ctx.deps.db.add_link(
            url=url,
            title=result.feed_title,
            tags=["ai"],
        )
        ctx.deps.db.update_availability(
            url=url,
            is_available=True,
            http_status=result.http_status,
            feed_title=result.feed_title,
            item_count=result.feed_item_count,
        )
        status_line = (
                f"✅ AVAILABLE | title='{result.feed_title}' | "
                f"items={result.feed_item_count} | "
                + ("⭐ NEW – added to DB" if is_new else "already in DB")
        )
    else:
        logger.warning(f"Feed {url} is unavailable: {result.error} (HTTP {result.http_status})")
        if ctx.deps.db.url_exists(url):
            ctx.deps.db.update_availability(
                url=url,
                is_available=False,
                http_status=result.http_status,
            )
        status_line = (
            f"❌ UNAVAILABLE | error={result.error} | "
            f"http={result.http_status}"
        )

    db_stats = ctx.deps.db.get_stats()
    return (
        f"URL    : {url}\n"
        f"Status : {status_line}\n"
        f"Time   : {result.check_duration_seconds:.2f}s\n"
        f"DB     : {db_stats['total']} total | "
        f"{db_stats['available']} available "
        f"({db_stats['availability_rate']:.1%} rate)"
    )


# ── MCP Tool 3: get links ───────────────────────────────────────────────────

@rss_agent.tool
async def get_rss_links(ctx: RunContext[AgentDeps], limit: int = 50) -> str:
    """
    MCP Tool – Get current RSS links from the database.

    Args:
        limit: Maximum number of links to return (default 50).

    Returns:
        A formatted list of RSS links with their status and metadata.
    """
    logger.info("Step: Getting RSS links from database (limit=%d)", limit)
    links = ctx.deps.db.get_all_links()[:limit]

    if not links:
        return "No RSS links in database."

    lines = [f"Current RSS Links (showing {len(links)}):", ""]
    for lnk in links:
        status = "✅" if lnk.is_available else "❌"
        title = f" [{lnk.title}]" if lnk.title else ""
        items = f" ({lnk.feed_item_count} items)" if lnk.feed_item_count is not None else ""
        lines.append(f"  {status} {lnk.url}{title}{items}")

    return "\n".join(lines)


# ── MCP Tool 4: delete link ─────────────────────────────────────────────────

@rss_agent.tool
async def delete_rss_link(ctx: RunContext[AgentDeps], url: str) -> str:
    """
    MCP Tool – Delete an RSS link from the database.

    Args:
        url: The RSS feed URL to delete.

    Returns:
        Confirmation message indicating success or failure.
    """
    logger.info("Step: Deleting RSS link: %s", url)
    deleted = ctx.deps.db.delete_link(url)

    if deleted:
        stats = ctx.deps.db.get_stats()
        return (
            f"✅ Deleted RSS link: {url}\n"
            f"DB now has {stats['total']} total links."
        )
    else:
        return f"❌ Link not found in database: {url}"


# ── MCP Tool 5: add link ────────────────────────────────────────────────────

@rss_agent.tool
async def add_rss_link(
        ctx: RunContext[AgentDeps], url: str, title: Optional[str] = None,
        tags: Optional[List[str]] = None
) -> str:
    """
    MCP Tool – Add a new RSS link to the database.

    Args:
        url: The RSS feed URL to add.
        title: Optional title for the feed.
        tags: Optional list of tags (default: ["ai"]).

    Returns:
        Confirmation message indicating if the link was added or already exists.
    """
    logger.info("Step: Adding RSS link: %s", url)
    is_new, link = ctx.deps.db.add_link(url=url, title=title, tags=tags or ["ai"])

    if is_new:
        stats = ctx.deps.db.get_stats()
        return (
            f"✅ Added new RSS link: {url}\n"
            f"Title: {link.title or 'N/A'}\n"
            f"DB now has {stats['total']} total links."
        )
    else:
        return f"ℹ️ Link already exists in database: {url}"


# ── High-level runner helpers ─────────────────────────────────────────────────

async def run_discovery(query: str, db: RSSDatabase) -> AgentRunResult:
    """Run the agent for a discovery query and return a structured result."""
    t0 = time.time()
    deps = AgentDeps(db=db)

    console.print(
        f"\n{_ts()} [bold cyan]🤖 AGENT RUN[/bold cyan] '{query}'"
    )
    logger.info(f"Starting agent discovery run with query: {query}")

    initial_total = db.get_stats()["total"]

    try:
        result = await rss_agent.run(query, deps=deps)
        response_text = str(result.data) if hasattr(result, "data") else str(result)
        logger.debug("Agent raw response: %s", response_text)
    except Exception:
        logger.exception("Agent failed during run")
        raise

    final_total = db.get_stats()["total"]
    duration = time.time() - t0

    # ── Print the agent response ─────────────────────────────────────────────
    console.print()
    console.print(f"[bold cyan]🤖 AGENT RESPONSE[/bold cyan] [dim]{query}[/dim]")
    console.print(result.output if hasattr(result, "output") else response_text)
    console.print()
    console.print(Rule(style="dim cyan"))

    console.print(
        f"{_ts()} [green]✅ Agent finished in {duration:.2f}s | "
        f"+{final_total - initial_total} new links[/green]"
    )

    return AgentRunResult(
        query=query,
        response=response_text,
        links_found=[lnk.url for lnk in db.get_all_links()],
        new_links_added=final_total - initial_total,
        duration_seconds=duration,
    )


async def run_and_return_links(
        query: str, db: Optional[RSSDatabase] = None
) -> List[str]:
    """Convenience wrapper used by the eval runner."""
    if db is None:
        db = RSSDatabase()
    await run_discovery(query, db)
    return [lnk.url for lnk in db.get_all_links()]
