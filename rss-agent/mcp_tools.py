"""
MCP Tool implementations for the RSS AI Agent.

Two tools are exposed:
  1. web_search_rss_links   – DuckDuckGo search for RSS feeds about AI.
  2. check_rss_availability – HTTP + feedparser validation of an RSS URL.

Both functions are pure (no agent state) so they can also be unit-tested
or called from the standalone MCP server (mcp_server.py).
"""
from __future__ import annotations

import re
import time
from datetime import datetime
from typing import List, Optional

import feedparser
import requests
from rich.console import Console

from models import RSSAvailabilityResult, RSSSearchResult
import config

# ── Logging Setup ─────────────────────────────────────────────────────────────

import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.WARNING,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=Console(stderr=True))],
)

# Silence noisy third-party loggers unless they are at least WARNING
logging.getLogger("pydantic_ai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger("rss_agent.mcp_tools")

console = Console()


def _ts() -> str:
    return f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim]"


# ── Tool 1: Web search for RSS links ──────────────────────────────────────────

def web_search_rss_links(
    query: str,
    max_results: int = config.MAX_SEARCH_RESULTS,
) -> RSSSearchResult:
    """
    MCP Tool – Search the web for AI-related RSS feed URLs.

    Uses DuckDuckGo (no API key required).  Falls back to the seed list when
    the library is unavailable or rate-limited.

    Returns a RSSSearchResult with unique, normalised URLs.
    """
    start = time.time()
    raw: List[str] = []

    console.print(f"{_ts()} [bold cyan]🔍 MCP:search_rss_links[/bold cyan] query='{query}'")
    logger.debug(f"Starting web search for: {query}")

    # ── primary: DuckDuckGo ───────────────────────────────────────────────────
    try:
        from ddgs import DDGS

        expanded = (
            f"{query} filetype:xml "
            "OR inurl:rss OR inurl:feed OR inurl:atom"
        )
        with DDGS() as ddgs:
            for r in ddgs.text(expanded, max_results=max_results):
                href = r.get("href", "")
                body = r.get("body", "")
                logger.debug(f"DDG Result: {href}")

                # Collect direct-match URLs
                if href and any(
                    kw in href.lower() for kw in ("rss", "feed", "atom", "xml")
                ):
                    raw.append(href)

                # Also mine URLs embedded in the snippet
                if body:
                    embedded = re.findall(
                        r"https?://[^\s<>\"']+(?:rss|feed|atom|xml)[^\s<>\"']*",
                        body,
                        re.IGNORECASE,
                    )
                    if embedded:
                        logger.debug(f"Extracted from snippet: {embedded}")
                    raw.extend(embedded)
                else:
                    logger.warning(f"No snippet body for {href}")

        console.print(
            f"{_ts()} [green]✅ DuckDuckGo returned {len(raw)} raw URLs[/green]"
        )

    except ImportError:
        console.print(
            f"{_ts()} [yellow]⚠️  duckduckgo-search not installed – using seed list[/yellow]"
        )
        raw = list(config.SEED_RSS_FEEDS)

    except Exception as exc:
        console.print(f"{_ts()} [red]❌ DuckDuckGo error: {exc} – using seed list[/red]")
        raw = list(config.SEED_RSS_FEEDS)

    # ── de-duplicate while preserving discovery order ─────────────────────────
    seen: set[str] = set()
    unique: List[str] = []
    for url in raw:
        key = url.strip().rstrip("/")
        if key and key not in seen:
            seen.add(key)
            unique.append(key)
            logger.debug(f"Unique URL kept: {key}")

    duration = time.time() - start
    console.print(
        f"{_ts()} [bold green]✅ MCP:search_rss_links[/bold green] "
        f"{len(unique)} unique URLs in {duration:.2f}s"
    )

    return RSSSearchResult(
        query=query,
        links_found=unique,
        search_duration_seconds=duration,
    )


# ── Tool 2: Check RSS link availability ──────────────────────────────────────

def check_rss_availability(
    url: str,
    timeout: int = config.REQUEST_TIMEOUT,
) -> RSSAvailabilityResult:
    """
    MCP Tool – Verify that an RSS feed URL is reachable and parseable.

    Steps:
      1. HTTP GET with a browser-like User-Agent.
      2. feedparser validation of the response body.
      3. Returns metadata: feed title, item count, HTTP status.
    """
    start = time.time()
    console.print(f"{_ts()} [bold magenta]🔧 MCP:check_rss_availability[/bold magenta] {url}")
    logger.debug(f"Checking URL: {url} with timeout {timeout}")

    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; RSSBot/1.0; "
                    "+https://github.com/ollama-dev)"
                ),
                "Accept": (
                    "application/rss+xml, application/atom+xml, "
                    "text/xml, application/xml, */*"
                ),
            },
            allow_redirects=True,
        )

        http_status = resp.status_code
        logger.debug(f"HTTP response for {url}: {http_status}")
        logger.debug(f"Response headers: {resp.headers}")

        if resp.status_code != 200:
            dur = time.time() - start
            console.print(
                f"{_ts()} [red]❌ HTTP {http_status}[/red] {url} ({dur:.2f}s)"
            )
            return RSSAvailabilityResult(
                url=url,
                is_available=False,
                http_status=http_status,
                error=f"HTTP {http_status}",
                check_duration_seconds=dur,
            )

        # Parse the feed
        logger.debug(f"Parsing feed content for {url} ({len(resp.content)} bytes)")
        feed = feedparser.parse(resp.content)

        if feed.bozo:
             logger.warning(f"Feedparser 'bozo' bit set for {url}: {feed.bozo_exception}")

        is_valid = bool(feed.feed.get("title") or feed.entries)
        feed_title: Optional[str] = feed.feed.get("title") or None
        item_count = len(feed.entries)
        dur = time.time() - start

        if is_valid:
            console.print(
                f"{_ts()} [green]✅ VALID[/green] '{feed_title}' "
                f"| {item_count} items | {dur:.2f}s"
            )
        else:
            console.print(
                f"{_ts()} [yellow]⚠️  HTTP 200 but no RSS content[/yellow] {url}"
            )

        return RSSAvailabilityResult(
            url=url,
            is_available=is_valid,
            http_status=http_status,
            feed_title=feed_title,
            feed_item_count=item_count,
            check_duration_seconds=dur,
        )

    except requests.Timeout:
        dur = time.time() - start
        console.print(f"{_ts()} [red]❌ TIMEOUT[/red] {url} ({dur:.2f}s)")
        return RSSAvailabilityResult(
            url=url,
            is_available=False,
            error="Timeout",
            check_duration_seconds=dur,
        )

    except Exception as exc:
        dur = time.time() - start
        console.print(f"{_ts()} [red]❌ ERROR[/red] {url}: {exc}")
        return RSSAvailabilityResult(
            url=url,
            is_available=False,
            error=str(exc),
            check_duration_seconds=dur,
        )
