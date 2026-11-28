"""Tests for main.py. Every step is faked, so the pipeline runs without network or keys."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import main
from src.collect import Article
from src.deliver import HEADER
from tests.fakes import FakeWorksheet, make_result


def make_article(i: int) -> Article:
    return Article(
        title=f"Title {i}",
        link=f"https://example.com/{i}",
        summary=f"Feed text {i}",
        source="Sample",
        published="2026-11-27",
    )


@pytest.fixture
def config_file(tmp_path: Path) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(
        'topic: "AI news"\ntop_n: 2\nfeeds:\n  - name: Sample\n    url: sample.xml\n',
        encoding="utf-8",
    )
    return str(path)


class Pipeline:
    """Replaces collect, analyze, open_worksheet and send_telegram inside main.py."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, articles: list[Article]) -> None:
        self.worksheet = FakeWorksheet()
        self.analyzed: list[Article] = []
        self.sent: list[str] = []
        self.fail_analysis = False

        monkeypatch.setattr(main, "collect", lambda config: list(articles))
        monkeypatch.setattr(main, "analyze", self.analyze)
        monkeypatch.setattr(main, "open_worksheet", lambda: self.worksheet)
        monkeypatch.setattr(main, "send_telegram", self.sent.append)

    def analyze(self, items: list[Article], topic: str) -> list[Any]:
        self.analyzed.extend(items)
        if self.fail_analysis:
            return []  # what analyze() returns when every batch was skipped
        number = {a["link"]: int(a["link"].rsplit("/", 1)[1]) for a in items}
        return [make_result(number[a["link"]], relevance=5) for a in items]


def test_full_run_saves_to_sheet_and_sends_top_n(
    monkeypatch: pytest.MonkeyPatch, config_file: str
) -> None:
    pipeline = Pipeline(monkeypatch, [make_article(i) for i in range(3)])

    assert main.run(config_file) == 0

    assert len(pipeline.worksheet.rows) == 1 + 3  # header + 3 articles
    [message] = pipeline.sent
    assert "Top 2 of 3 new articles" in message


def test_known_links_are_not_analyzed_again(
    monkeypatch: pytest.MonkeyPatch, config_file: str
) -> None:
    pipeline = Pipeline(monkeypatch, [make_article(0), make_article(1)])
    pipeline.worksheet.rows = [HEADER, ["2026-11-27", "", "", "", 3, "", "https://example.com/0"]]

    assert main.run(config_file) == 0

    assert [a["link"] for a in pipeline.analyzed] == ["https://example.com/1"]
    assert "Top 1 of 1 new articles" in pipeline.sent[0]


def test_nothing_new_sends_short_message_without_calling_llm(
    monkeypatch: pytest.MonkeyPatch, config_file: str
) -> None:
    pipeline = Pipeline(monkeypatch, [make_article(0)])
    pipeline.worksheet.rows = [HEADER, ["2026-11-27", "", "", "", 3, "", "https://example.com/0"]]

    assert main.run(config_file) == 0

    assert pipeline.analyzed == []
    assert "No new articles today." in pipeline.sent[0]


def test_dry_run_prints_message_and_does_not_touch_sheet_or_telegram(
    monkeypatch: pytest.MonkeyPatch, config_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    pipeline = Pipeline(monkeypatch, [make_article(0)])

    def no_sheet() -> None:
        raise AssertionError("dry run must not open Google Sheets")

    monkeypatch.setattr(main, "open_worksheet", no_sheet)

    assert main.run(config_file, dry_run=True) == 0

    assert "1. Title 0 [5/5]" in capsys.readouterr().out
    assert pipeline.sent == []


def test_run_fails_when_no_articles_are_collected(
    monkeypatch: pytest.MonkeyPatch, config_file: str
) -> None:
    pipeline = Pipeline(monkeypatch, [])
    assert main.run(config_file) == 1
    assert pipeline.sent == []


def test_run_fails_and_sends_nothing_when_analysis_fails(
    monkeypatch: pytest.MonkeyPatch, config_file: str
) -> None:
    pipeline = Pipeline(monkeypatch, [make_article(0)])
    pipeline.fail_analysis = True

    assert main.run(config_file) == 1

    assert pipeline.sent == []
    assert pipeline.worksheet.rows == []


def test_main_returns_2_on_missing_settings(
    monkeypatch: pytest.MonkeyPatch, config_file: str
) -> None:
    monkeypatch.setattr(main, "collect", lambda config: [make_article(0)])
    monkeypatch.delenv("GOOGLE_CREDENTIALS", raising=False)
    monkeypatch.setattr(main, "load_dotenv", lambda: None)  # ignore a real .env on this machine
    monkeypatch.setattr("sys.argv", ["main.py", "--config", config_file])

    assert main.main() == 2
