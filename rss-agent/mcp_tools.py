"""
MCP Tool implementations for the RSS AI Agent.

Two tools are exposed:
  1. web_search_rss_links   – DuckDuckGo search for RSS feeds about AI.
  2. check_rss_availability – HTTP + feedparser validation of an RSS URL.

Both functions are pure (no agent state) so they can also be unit-tested
or called from the standalone MCP server (mcp_server.py).
"""
from __future__ import annotations

import logging
import re
import time
import random
from datetime import datetime
from typing import List, Optional

import feedparser
import requests
from rich.console import Console
from rich.logging import RichHandler

import config
from models import RSSAvailabilityResult, RSSSearchResult

# ── Logging Setup ─────────────────────────────────────────────────────────────

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO,
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


# ── Random browser impersonation ──────────────────────────────────────────────

_USER_AGENTS: List[str] = [
    # Windows 11 – Chrome 124
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Windows 11 – Chrome 122
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.6261.112 Safari/537.36",
    # Windows 11 – Edge 124
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    # Windows 10 – Firefox 125
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    # Windows 10 – Firefox 122
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
    # macOS 14 Sonoma – Chrome 124
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # macOS 14 – Safari 17
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    # macOS 13 – Firefox 124
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13.6; rv:124.0) Gecko/20100101 Firefox/124.0",
    # Linux Ubuntu – Chrome 124
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Linux – Firefox 123
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
    # Linux Fedora – Chrome 122
    "Mozilla/5.0 (X11; Fedora; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]

_ACCEPT_LANGUAGES: List[str] = [
    "en-US,en;q=0.9",
    "en-GB,en;q=0.9",
    "en-US,en;q=0.8,de;q=0.6",
    "en-US,en;q=0.9,fr;q=0.7",
    "en-CA,en;q=0.9",
    "en-AU,en;q=0.9",
]

_CACHE_CONTROLS: List[str] = [
    "no-cache",
    "max-age=0",
    "no-cache, no-store",
]


def _random_headers() -> Dict[str, str]:
    """Return a dict of HTTP headers that impersonate a random real browser."""
    ua = random.choice(_USER_AGENTS)
    is_firefox = "Firefox" in ua
    is_safari = "Safari" in ua and "Chrome" not in ua

    accept_rss = (
        "application/rss+xml, application/atom+xml, "
        "text/xml, application/xml, */*;q=0.8"
    )
    accept_html = (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
        if is_firefox
        else "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    )

    headers: Dict[str, str] = {
        "User-Agent": ua,
        "Accept": accept_rss if random.random() < 0.6 else accept_html,
        "Accept-Language": random.choice(_ACCEPT_LANGUAGES),
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control": random.choice(_CACHE_CONTROLS),
        "Connection": "keep-alive",
    }

    # DNT header – not every browser sends it
    if random.random() < 0.4:
        headers["DNT"] = "1"

    # Upgrade-Insecure-Requests (Chromium / Firefox)
    if not is_safari:
        headers["Upgrade-Insecure-Requests"] = "1"

    # Sec-Fetch-* headers – only Chromium-family sends these
    if not is_firefox and not is_safari:
        headers["Sec-Fetch-Dest"] = "document"
        headers["Sec-Fetch-Mode"] = "navigate"
        headers["Sec-Fetch-Site"] = random.choice(["none", "cross-site"])
        headers["Sec-Fetch-User"] = "?1"
        headers["Sec-CH-UA"] = '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"'
        headers["Sec-CH-UA-Mobile"] = "?0"
        headers["Sec-CH-UA-Platform"] = (
            '"Windows"' if "Windows" in ua else
            '"macOS"' if "Macintosh" in ua else '"Linux"'
        )

    logger.debug(f"Using UA: {ua}")
    return headers


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
    headers = _random_headers()
    ua_short = headers["User-Agent"].split(")")[0].split("(")[-1]  # e.g. "Windows NT 10.0; Win64; x64"
    console.print(
        f"{_ts()} [bold magenta]🔧 MCP:check_rss_availability[/bold magenta] "
        f"{url} [dim]({ua_short})[/dim]"
    )
    logger.debug(f"Checking URL: {url} with timeout {timeout}")

    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers=headers,
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
