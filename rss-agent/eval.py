"""
Evaluation framework for the RSS AI Agent.

Provides:
  - load_eval_cases()     – load EvalCase objects from config.EVAL_CASES
  - score_result()        – compute a 0.0–1.0 score for one result
  - EvalRunner            – async runner that feeds cases into the agent
  - print_eval_result()   – rich single-result display
  - print_eval_stats()    – rich aggregated stats panel
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config
from models import EvalCase, EvalResult, EvalStats

console = Console()

_EVAL_PATH = Path(__file__).parent / config.EVAL_RESULTS_FILE


def _ts() -> str:
    return f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim]"


# ── Case loading ──────────────────────────────────────────────────────────────

def load_eval_cases() -> List[EvalCase]:
    """Instantiate EvalCase objects from the dicts in config.EVAL_CASES."""
    return [EvalCase(**d) for d in config.EVAL_CASES]


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_result(
    found_links: List[str],
    case: EvalCase,
    duration: float,
) -> tuple[float, List[str], List[str]]:
    """
    Return (score, matched_patterns, unmatched_patterns).

    Score breakdown:
      60 % – pattern coverage (how many expected patterns were found)
      30 % – link count      (found >= min_links_expected)
      10 % – timing          (within max_duration_seconds)
    """
    # Pattern coverage
    matched: List[str] = []
    unmatched: List[str] = []
    if case.expected_url_patterns:
        for pat in case.expected_url_patterns:
            if any(pat.lower() in lnk.lower() for lnk in found_links):
                matched.append(pat)
            else:
                unmatched.append(pat)
        pattern_score = len(matched) / len(case.expected_url_patterns)
    else:
        pattern_score = 1.0  # no patterns → full marks on this axis

    # Count score
    count_score = (
        min(1.0, len(found_links) / case.min_links_expected)
        if case.min_links_expected > 0
        else 1.0
    )

    # Timing score
    if duration <= case.max_duration_seconds:
        time_score = 1.0
    else:
        overshoot = duration - case.max_duration_seconds
        time_score = max(0.0, 1.0 - overshoot / case.max_duration_seconds)

    final = pattern_score * 0.6 + count_score * 0.3 + time_score * 0.1
    return round(final, 4), matched, unmatched


# ── Persistence helpers ───────────────────────────────────────────────────────

def _save_results(results: List[EvalResult]) -> None:
    payload = [r.model_dump(mode="json") for r in results]
    with open(_EVAL_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)  # type: ignore[arg-type]


def _load_results() -> List[EvalResult]:
    if not _EVAL_PATH.exists():
        return []
    try:
        with open(_EVAL_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return [EvalResult(**r) for r in data]
    except Exception:
        return []


# ── Display helpers ───────────────────────────────────────────────────────────

def print_eval_result(result: EvalResult) -> None:
    status = (
        "[bold green]PASS ✅[/bold green]"
        if result.passed
        else "[bold red]FAIL ❌[/bold red]"
    )
    console.print(
        f"{_ts()} [cyan]Eval[/cyan] {result.case_name} → {status} "
        f"score={result.score:.2f} dur={result.duration_seconds:.1f}s"
    )
    if result.matched_patterns:
        console.print(
            f"  [green]✓ matched: {', '.join(result.matched_patterns)}[/green]"
        )
    if result.unmatched_patterns:
        console.print(
            f"  [red]✗ missing: {', '.join(result.unmatched_patterns)}[/red]"
        )
    if result.error:
        console.print(f"  [red]error: {result.error}[/red]")


def print_eval_stats(stats: EvalStats) -> None:
    tbl = Table.grid(padding=(0, 2))
    tbl.add_column(justify="right", style="cyan", no_wrap=True)
    tbl.add_column(justify="left", style="white")

    rate_colour = "green" if stats.pass_rate >= 0.7 else "red"

    tbl.add_row("", "[bold yellow]═══ EVAL STATISTICS ═══[/bold yellow]")
    tbl.add_row("📊 Total Runs:", str(stats.total_runs))
    tbl.add_row("✅ Passed:", f"[green]{stats.total_passed}[/green]")
    tbl.add_row("❌ Failed:", f"[red]{stats.total_failed}[/red]")
    tbl.add_row(
        "📈 Pass Rate:",
        f"[{rate_colour}]{stats.pass_rate:.1%}[/{rate_colour}]",
    )
    tbl.add_row("⭐ Avg Score:", f"{stats.avg_score:.3f}")
    tbl.add_row("⏱️  Avg Duration:", f"{stats.avg_duration_seconds:.1f}s")
    tbl.add_row("🔗 Total Links:", str(stats.total_links_found))
    tbl.add_row("🌐 Unique Links:", str(stats.unique_links_found))
    tbl.add_row("📡 Avail. Rate:", f"{stats.availability_rate:.1%}")
    tbl.add_row(
        "🕒 Last Updated:",
        stats.last_updated.strftime("%Y-%m-%d %H:%M:%S"),
    )

    console.print(
        Panel(tbl, title="[bold cyan]📊 EVAL REPORT[/bold cyan]", border_style="cyan")
    )


# ── Eval runner ───────────────────────────────────────────────────────────────

# Type alias: an async function that takes a query and returns a list of URLs
AgentRunFn = Callable[[str], Awaitable[List[str]]]


class EvalRunner:
    """
    Runs evaluation cases against a callable agent function and accumulates stats.

    Usage::

        runner = EvalRunner(my_agent_fn)
        stats  = await runner.run_all()
        # or run a single case:
        result = await runner.run_case(cases[0])
    """

    def __init__(self, agent_run_fn: AgentRunFn) -> None:
        self._run = agent_run_fn
        self.cases: List[EvalCase] = load_eval_cases()
        self.results: List[EvalResult] = _load_results()

    # ── single case ──────────────────────────────────────────────────────────

    async def run_case(self, case: EvalCase) -> EvalResult:
        console.print(
            f"\n{_ts()} [bold cyan]▶ eval case:[/bold cyan] {case.name}"
        )
        console.print(f"  [dim]{case.description}[/dim]")
        console.print(f"  [dim]query: {case.query}[/dim]")

        t0 = time.time()
        error: str | None = None
        found_links: List[str] = []

        try:
            found_links = await self._run(case.query)
        except Exception as exc:
            error = str(exc)
            console.print(f"{_ts()} [red]❌ agent raised: {exc}[/red]")

        duration = time.time() - t0
        score, matched, unmatched = score_result(found_links, case, duration)
        passed = score >= 0.5 and error is None

        result = EvalResult(
            eval_case_id=case.id,
            case_name=case.name,
            passed=passed,
            score=score,
            found_links=found_links,
            matched_patterns=matched,
            unmatched_patterns=unmatched,
            duration_seconds=duration,
            error=error,
        )

        self.results.append(result)
        _save_results(self.results)
        print_eval_result(result)
        return result

    # ── full suite ────────────────────────────────────────────────────────────

    async def run_all(self) -> EvalStats:
        console.print(
            f"\n{_ts()} [bold yellow]🧪 Starting eval suite "
            f"({len(self.cases)} cases)[/bold yellow]"
        )
        run_results: List[EvalResult] = []
        for case in self.cases:
            run_results.append(await self.run_case(case))

        stats = EvalStats()
        stats.update_from_results(run_results)
        print_eval_stats(stats)
        return stats

    # ── historical stats ──────────────────────────────────────────────────────

    def get_historical_stats(self) -> EvalStats:
        stats = EvalStats()
        if self.results:
            stats.update_from_results(self.results)
        return stats


