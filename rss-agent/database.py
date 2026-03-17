"""
Persistent JSON database for unique RSS feed links.
Thread-safe via a simple threading.Lock.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

from models import RSSLink

_DEFAULT_DB = Path(__file__).parent / "rss_db.json"
logger = logging.getLogger("rss_agent.database")


class RSSDatabase:
    """
    JSON-backed store for RSSLink objects.
    The primary key is the normalised URL (trailing slash stripped, lower-cased scheme).
    """

    def __init__(self, db_path: str | Path = _DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._links: Dict[str, RSSLink] = {}
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _normalise(self, url: str) -> str:
        cleaned = url.strip()
        if not cleaned:
            return ""

        # Normalize scheme/host case and remove trailing slashes for stable dedupe keys.
        parsed = urlsplit(cleaned)
        if parsed.scheme:
            path = parsed.path.rstrip("/")
            return urlunsplit(
                (
                    parsed.scheme.lower(),
                    parsed.netloc.lower(),
                    path,
                    parsed.query,
                    parsed.fragment,
                )
            )

        return cleaned.rstrip("/")

    def _load(self) -> None:
        if not self.db_path.exists():
            logger.debug("Database file does not exist yet: %s", self.db_path)
            return
        try:
            with open(self.db_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for item in data.get("links", []):
                link = RSSLink(**item)
                self._links[self._normalise(link.url)] = link
            logger.info("Loaded database from %s (%d links)", self.db_path, len(self._links))
        except Exception as exc:
            logger.warning("Could not load %s: %s", self.db_path, exc)

    def save(self) -> None:
        with self._lock:
            payload = {
                "links": [lnk.model_dump(mode="json") for lnk in self._links.values()],
                "last_saved": datetime.now().isoformat(),
                "total_links": len(self._links),
            }
        try:
            logger.debug("Saving database to %s", self.db_path)
            with open(self.db_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)  # type: ignore[arg-type]
            logger.info("Database saved: %d total links", payload["total_links"])
        except Exception:
            logger.exception("Failed to save database to %s", self.db_path)
            raise

    # ── lookups/stats ─────────────────────────────────────────────────────────

    def url_exists(self, url: str) -> bool:
        key = self._normalise(url)
        with self._lock:
            return key in self._links

    def get_all_links(self) -> List[RSSLink]:
        with self._lock:
            return list(self._links.values())

    def get_links(self) -> List[RSSLink]:
        """Get all current links in the database."""
        with self._lock:
            return list(self._links.values())

    def get_stats(self) -> Dict[str, object]:
        with self._lock:
            links = list(self._links.values())

        total = len(links)
        available = sum(1 for lnk in links if lnk.is_available)
        unavailable = total - available
        checked_links = sum(1 for lnk in links if lnk.last_checked is not None)
        never_checked = total - checked_links
        availability_rate = (available / total) if total else 0.0

        status_counts: Dict[int, int] = {}
        last_checked_values = [lnk.last_checked for lnk in links if lnk.last_checked is not None]
        for lnk in links:
            if lnk.http_status is not None:
                status_counts[lnk.http_status] = status_counts.get(lnk.http_status, 0) + 1

        stats: Dict[str, object] = {
            "total": total,
            "available": available,
            "unavailable": unavailable,
            "availability_rate": availability_rate,
            "checked_links": checked_links,
            "never_checked": never_checked,
            "http_status_counts": dict(sorted(status_counts.items())),
            "last_checked": max(last_checked_values) if last_checked_values else None,
        }
        logger.debug("Computed DB stats: %s", stats)
        return stats

    # ── mutations ─────────────────────────────────────────────────────────────

    def add_link(
        self,
        url: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Tuple[bool, RSSLink]:
        """
        Insert a new link.  Returns (is_new, link).
        If the URL already exists, the existing record is returned unchanged.
        """
        key = self._normalise(url)
        if not key:
            raise ValueError("URL cannot be empty")

        with self._lock:
            if key in self._links:
                logger.debug("Link already exists: %s", key)
                return False, self._links[key]
            link = RSSLink(url=key, title=title, description=description, tags=tags or [])
            self._links[key] = link
        logger.info("Added new link: %s", key)
        self.save()
        return True, link

    def update_availability(
        self,
        url: str,
        is_available: bool,
        http_status: Optional[int] = None,
        feed_title: Optional[str] = None,
        item_count: Optional[int] = None,
    ) -> None:
        key = self._normalise(url)
        with self._lock:
            if key not in self._links:
                logger.warning("Attempted to update availability for non-existent link: %s", key)
                return
            lnk = self._links[key]
            lnk.is_available = is_available
            lnk.http_status = http_status
            lnk.last_checked = datetime.now()
            lnk.check_count += 1
            if feed_title:
                lnk.title = feed_title
            if item_count is not None:
                lnk.feed_item_count = item_count
        logger.info(
            "Updated availability for %s: available=%s status=%s items=%s",
            key,
            is_available,
            http_status,
            item_count,
        )
        self.save()

    def delete_link(self, url: str) -> bool:
        """
        Delete a link from the database by URL.
        Returns True if the link was found and deleted, False otherwise.
        """
        key = self._normalise(url)
        with self._lock:
            if key not in self._links:
                logger.warning("Attempted to delete non-existent link: %s", key)
                return False
            del self._links[key]
        logger.info("Deleted link: %s", key)
        self.save()
        return True
