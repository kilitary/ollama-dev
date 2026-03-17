"""
Pydantic data models for the RSS AI Agent.
Covers feed links, MCP tool results, eval cases, and eval statistics.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── RSS Link ──────────────────────────────────────────────────────────────────

class RSSLink(BaseModel):
    """Represents a single discovered RSS feed entry in the database."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    url: str
    title: Optional[str] = None
    description: Optional[str] = None
    language: Optional[str] = None
    category: str = "ai"
    tags: List[str] = Field(default_factory=list)

    discovered_at: datetime = Field(default_factory=datetime.now)
    last_checked: Optional[datetime] = None
    is_available: bool = True
    check_count: int = 0
    http_status: Optional[int] = None
    feed_item_count: Optional[int] = None

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


# ── MCP Tool result models ────────────────────────────────────────────────────

class RSSSearchResult(BaseModel):
    """Returned by the `search_rss_links` MCP tool."""

    query: str
    links_found: List[str] = Field(default_factory=list)
    new_links: List[str] = Field(default_factory=list)
    duplicate_links: List[str] = Field(default_factory=list)
    search_duration_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


class RSSAvailabilityResult(BaseModel):
    """Returned by the `check_rss_availability` MCP tool."""

    url: str
    is_available: bool = False
    http_status: Optional[int] = None
    error: Optional[str] = None
    feed_title: Optional[str] = None
    feed_item_count: Optional[int] = None
    check_duration_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


# ── Evaluation models ─────────────────────────────────────────────────────────

class EvalCase(BaseModel):
    """Defines a single evaluation test case for the agent."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    query: str
    # Substrings that should appear in at least one of the found URLs
    expected_url_patterns: List[str] = Field(default_factory=list)
    min_links_expected: int = 1
    max_duration_seconds: float = 60.0


class EvalResult(BaseModel):
    """Result of running one evaluation case."""

    eval_case_id: str
    case_name: str
    passed: bool
    score: float          # 0.0 – 1.0
    found_links: List[str] = Field(default_factory=list)
    matched_patterns: List[str] = Field(default_factory=list)
    unmatched_patterns: List[str] = Field(default_factory=list)
    duration_seconds: float = 0.0
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class EvalStats(BaseModel):
    """Aggregated statistics across multiple evaluation runs."""

    total_runs: int = 0
    total_passed: int = 0
    total_failed: int = 0
    pass_rate: float = 0.0
    avg_score: float = 0.0
    avg_duration_seconds: float = 0.0
    total_links_found: int = 0
    unique_links_found: int = 0
    availability_rate: float = 0.0
    last_updated: datetime = Field(default_factory=datetime.now)

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}

    def update_from_results(self, results: List[EvalResult]) -> None:
        if not results:
            return
        self.total_runs = len(results)
        self.total_passed = sum(1 for r in results if r.passed)
        self.total_failed = self.total_runs - self.total_passed
        self.pass_rate = self.total_passed / self.total_runs
        self.avg_score = sum(r.score for r in results) / self.total_runs
        self.avg_duration_seconds = sum(r.duration_seconds for r in results) / self.total_runs
        self.total_links_found = sum(len(r.found_links) for r in results)
        all_links: set[str] = set()
        for r in results:
            all_links.update(r.found_links)
        self.unique_links_found = len(all_links)
        self.last_updated = datetime.now()


# ── Agent run result ──────────────────────────────────────────────────────────

class AgentRunResult(BaseModel):
    """Summary of a single agent invocation."""

    query: str
    response: str
    links_found: List[str] = Field(default_factory=list)
    new_links_added: int = 0
    duration_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}

