"""Tests for src/collect.py. They read local sample feeds, so no network is needed."""

from __future__ import annotations

from pathlib import Path

import pytest

from src import collect as collect_module
from src.collect import clean_text, collect, load_config, parse_feed

DATA_DIR = Path(__file__).parent / "data"
FEED_1 = str(DATA_DIR / "sample_feed.xml")
FEED_2 = str(DATA_DIR / "sample_feed_2.xml")


def test_clean_text_removes_html_and_extra_spaces() -> None:
    assert clean_text("<p>Hello <b>world</b> &amp;\n\n friends</p>") == "Hello world & friends"


def test_parse_feed_reads_articles_and_skips_entries_without_link() -> None:
    articles = parse_feed(FEED_1, source="Sample")

    assert [a["title"] for a in articles] == [
        "New open-source LLM runs on a laptop",
        "Small shops automate invoices with AI",
        "Chip makers report record quarter",
    ]
    first = articles[0]
    assert first["link"] == "https://example.com/articles/local-llm"
    assert first["summary"] == "A new model fits in 8 GB of RAM & runs offline."
    assert first["source"] == "Sample"
    assert first["published"] == "2026-11-23"


def test_parse_feed_handles_missing_summary_and_date() -> None:
    chips = parse_feed(FEED_1, source="Sample")[-1]
    assert chips["summary"] == ""
    assert chips["published"] == ""


def test_parse_feed_respects_limit() -> None:
    assert len(parse_feed(FEED_1, source="Sample", limit=1)) == 1


def test_parse_feed_reads_atom_feeds() -> None:
    articles = parse_feed(FEED_2, source="Atom")
    assert articles[1]["link"] == "https://example.org/agents-support"
    assert articles[1]["published"] == "2026-11-25"


def test_parse_feed_returns_empty_list_for_unreadable_feed(tmp_path: Path) -> None:
    broken = tmp_path / "broken.xml"
    broken.write_text("this is not a feed", encoding="utf-8")
    assert parse_feed(str(broken), source="Broken") == []


def test_collect_merges_feeds_and_removes_duplicate_links() -> None:
    config = {"feeds": [{"name": "One", "url": FEED_1}, {"name": "Two", "url": FEED_2}]}

    articles = collect(config)

    links = [a["link"] for a in articles]
    assert len(links) == len(set(links)) == 4
    # The duplicate keeps the version from the first feed.
    invoices = next(a for a in articles if "invoices" in a["link"])
    assert invoices["source"] == "One"


def test_collect_skips_a_feed_that_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    real_parse_feed = collect_module.parse_feed

    def flaky_parse_feed(url: str, source: str, limit: int = 10):
        if source == "Down":
            raise ConnectionError("site is down")
        return real_parse_feed(url, source, limit)

    monkeypatch.setattr(collect_module, "parse_feed", flaky_parse_feed)
    config = {"feeds": [{"name": "Down", "url": "https://down.example"}, {"url": FEED_2}]}

    articles = collect(config)

    assert len(articles) == 2
    assert articles[0]["source"] == FEED_2  # no name given -> the URL is used as the source


def test_load_config_reads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "topic: AI\nlimit_per_feed: 3\nfeeds:\n  - name: HN\n    url: https://hnrss.org/frontpage\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config["topic"] == "AI"
    assert config["limit_per_feed"] == 3
    assert config["feeds"][0]["url"] == "https://hnrss.org/frontpage"


@pytest.mark.parametrize(
    "content",
    ["topic: AI\n", "feeds: []\n", "feeds:\n  - name: No URL\n"],
)
def test_load_config_rejects_missing_feeds(tmp_path: Path, content: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_project_config_is_valid() -> None:
    config = load_config(Path(__file__).parent.parent / "config.yaml")
    assert config["topic"]
    assert all(feed["url"].startswith("https://") for feed in config["feeds"])
