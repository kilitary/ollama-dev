"""
RSS AI Agent – CLI entry point.

Modes
-----
  discover  Run the agent to find new AI RSS feeds  (default)
  eval      Run the full evaluation suite and print stats
  seed      Pre-populate the DB with known AI feeds from config
  status    Print current DB stats + last N links
  history   Show historical eval statistics

Examples
--------
  python run.py
  python run.py --mode discover --query "AI newsletter RSS 2024"
  python run.py --mode eval
  python run.py --mode seed
  python run.py --mode status
"""
from __future__ import annotations

import asyncio
import argparse
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.utils import shuffle

# Ensure the rss-agent package dir is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent))

import config
from agent import run_and_return_links, run_discovery
from database import RSSDatabase
from eval import EvalRunner, EvalStats, print_eval_stats

console = Console()


def _ts() -> str:
    return f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim]"


# ── Display helpers ───────────────────────────────────────────────────────────

def print_banner() -> None:
    console.print(
        Panel(
            "[bold cyan]🤖 RSS AI AGENT[/bold cyan]\n"
            "[dim]Discovers · Verifies · Persists AI RSS feeds "
            "using PydanticAI + MCP tools[/dim]",
            border_style="cyan",
        )
    )


def print_db_status(db: RSSDatabase, last_n: int = 10) -> None:
    stats = db.get_stats()
    tbl = Table.grid(padding=(0, 2))
    tbl.add_column(justify="right", style="cyan", no_wrap=True)
    tbl.add_column(justify="left", style="white")

    tbl.add_row("", "[bold yellow]═══ DATABASE STATUS ═══[/bold yellow]")
    tbl.add_row("🔗 Total Links  :", str(stats["total"]))
    tbl.add_row("✅ Available    :", f"[green]{stats['available']}[/green]")
    tbl.add_row("❌ Unavailable  :", f"[red]{stats['unavailable']}[/red]")
    tbl.add_row("📡 Avail. Rate  :", f"{stats['availability_rate']:.1%}")

    console.print(
        Panel(tbl, title="[bold cyan]📊 DB STATUS[/bold cyan]", border_style="cyan")
    )

    links = db.get_all_links()
    if links:
        console.print(
            f"\n[cyan]Last {last_n} entries:[/cyan]"
        )
        for lnk in links[-last_n:]:
            icon = "✅" if lnk.is_available else "❌"
            title = f" [{lnk.title}]" if lnk.title else ""
            console.print(f"  {icon} {lnk.url[:90]}{title}")


# ── Mode handlers ─────────────────────────────────────────────────────────────

async def _seed(db: RSSDatabase) -> None:
    console.print(
        f"{_ts()} [cyan]📦 Seeding DB with "
        f"{len(config.SEED_RSS_FEEDS)} known feeds…[/cyan]"
    )
    added = 0
    for url in config.SEED_RSS_FEEDS:
        is_new, _ = db.add_link(url, tags=["ai", "seed"])
        if is_new:
            added += 1
    console.print(
        f"{_ts()} [green]✅ Seeded: {added} new | "
        f"{db.get_stats()['total']} total in DB[/green]"
    )


async def _discover(db: RSSDatabase, query: str | None) -> None:
    queries = [query] if query else shuffle(config.AI_RSS_SEARCH_QUERIES)
    for q in queries:
        await run_discovery(q, db)
    print_db_status(db)


async def _eval(db: RSSDatabase) -> "EvalStats":  # type: ignore[return]
    async def _agent_fn(q: str):
        return await run_and_return_links(q, db)

    runner = EvalRunner(_agent_fn)
    stats = await runner.run_all()
    return stats


async def _history(db: RSSDatabase) -> None:
    async def _agent_fn(q: str):  # dummy – won't be called
        return []

    runner = EvalRunner(_agent_fn)
    stats = runner.get_historical_stats()
    print_eval_stats(stats)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="RSS AI Agent – discover and maintain AI RSS feeds"
    )
    parser.add_argument(
        "--mode",
        choices=["discover", "eval", "seed", "status", "history"],
        default="discover",
        help="Operation mode (default: discover)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Custom search query for discover mode",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to custom rss_db.json file",
    )
    args = parser.parse_args()

    print_banner()

    db = RSSDatabase(args.db) if args.db else RSSDatabase()
    console.print(
        f"{_ts()} [cyan]ℹ️  DB loaded: "
        f"{db.get_stats()['total']} existing links[/cyan]"
    )

    if args.mode == "seed":
        await _seed(db)
        print_db_status(db)

    elif args.mode == "status":
        print_db_status(db, last_n=20)

    elif args.mode == "history":
        await _history(db)

    elif args.mode == "eval":
        await _eval(db)
        print_db_status(db)

    else:  # discover
        await _discover(db, args.query)


if __name__ == "__main__":
    asyncio.run(main())



