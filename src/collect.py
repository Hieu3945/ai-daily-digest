"""Collect articles from the RSS/Atom feeds listed in config.yaml.

RSS is used instead of scraping HTML because it is stable, fast and is the way
sites intend their content to be read by programs.

Run `python -m src.collect` to print what would be collected today.
"""

from __future__ import annotations

import html
import logging
import re
from pathlib import Path
from time import strftime
from typing import Any, TypedDict

import feedparser
import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config.yaml")
MAX_SUMMARY_CHARS = 1000  # keep prompts short: the LLM only needs the gist

_TAG_RE = re.compile(r"<[^>]+>")


class Article(TypedDict):
    """One collected article, ready to be sent to the LLM."""

    title: str
    link: str
    summary: str
    source: str
    published: str  # "YYYY-MM-DD", or "" if the feed has no date


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read the YAML config and check that it lists at least one feed."""
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    feeds = config.get("feeds")
    if not feeds:
        raise ValueError(f"{path}: 'feeds' must contain at least one feed.")
    for feed in feeds:
        if not isinstance(feed, dict) or not feed.get("url"):
            raise ValueError(f"{path}: every feed needs a 'url' (got {feed!r}).")
    return config


def clean_text(text: str) -> str:
    """Remove HTML tags and entities and collapse whitespace."""
    text = html.unescape(_TAG_RE.sub(" ", text))
    return " ".join(text.split())


def _published_date(entry: Any) -> str:
    """Return the entry's date as YYYY-MM-DD, or "" if the feed gives none."""
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    return strftime("%Y-%m-%d", parsed) if parsed else ""


def parse_feed(url: str, source: str, limit: int = 10) -> list[Article]:
    """Read one feed and return up to `limit` articles.

    `url` can be a web address or a local file path (handy for tests).
    Entries without a title or link are skipped.
    """
    feed = feedparser.parse(url)
    if feed.bozo and not feed.entries:
        # bozo = feedparser could not read the feed properly
        logger.warning("Could not read feed %s: %s", source, feed.get("bozo_exception"))
        return []

    articles: list[Article] = []
    for entry in feed.entries[:limit]:
        title = clean_text(entry.get("title", ""))
        link = entry.get("link", "").strip()
        if not title or not link:
            continue
        articles.append(
            Article(
                title=title,
                link=link,
                summary=clean_text(entry.get("summary", ""))[:MAX_SUMMARY_CHARS],
                source=source,
                published=_published_date(entry),
            )
        )
    return articles


def collect(config: dict[str, Any]) -> list[Article]:
    """Collect articles from every feed in the config, without duplicate links.

    One broken feed never stops the run: it is logged and skipped.
    """
    limit = int(config.get("limit_per_feed", 10))
    seen_links: set[str] = set()
    articles: list[Article] = []

    for feed in config["feeds"]:
        source = feed.get("name") or feed["url"]
        try:
            items = parse_feed(feed["url"], source=source, limit=limit)
        except Exception:  # network errors, bad XML, ... - keep going
            logger.exception("Failed to collect feed %s", source)
            continue

        for item in items:
            if item["link"] not in seen_links:
                seen_links.add(item["link"])
                articles.append(item)
        logger.info("%s: %d articles", source, len(items))

    return articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    collected = collect(load_config())
    print(f"Collected {len(collected)} articles")
    for article in collected[:10]:
        print(f"- [{article['source']}] {article['title']}")
