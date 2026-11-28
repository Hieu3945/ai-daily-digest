"""Tests for src/deliver.py. Google Sheets and Telegram are faked: no network or keys needed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import requests

from src import deliver
from src.deliver import (
    HEADER,
    MissingSettingError,
    TelegramError,
    format_message,
    keep_new,
    load_credentials,
    save_to_sheet,
    saved_links,
    send_telegram,
)
from tests.fakes import FakeWorksheet, make_result

TODAY = "2026-11-28"


# ---- Google Sheets -----------------------------------------------------------------------


def test_save_to_sheet_writes_header_and_rows_to_an_empty_sheet() -> None:
    ws = FakeWorksheet()

    added = save_to_sheet(ws, [make_result(0, relevance=5), make_result(1)], today=TODAY)

    assert added == 2
    assert ws.rows[0] == HEADER
    assert ws.rows[1] == [
        TODAY,
        "Title 0",
        "Sample",
        "Research",
        5,
        "Summary 0.",
        "https://example.com/0",
    ]
    assert set(ws.input_options) == {"RAW"}  # text is never run as a formula


def test_save_to_sheet_skips_links_that_are_already_saved() -> None:
    ws = FakeWorksheet()
    save_to_sheet(ws, [make_result(0)], today="2026-11-27")

    added = save_to_sheet(ws, [make_result(0), make_result(1)], today=TODAY)

    assert added == 1
    assert [row[-1] for row in ws.rows[1:]] == ["https://example.com/0", "https://example.com/1"]
    assert ws.rows.count(HEADER) == 1  # header is written only once


def test_saved_links_ignores_header_and_empty_cells() -> None:
    ws = FakeWorksheet([HEADER, [TODAY, "t", "s", "c", 3, "sum", "https://a.com"], [TODAY]])
    assert saved_links(ws) == {"https://a.com"}


def test_keep_new_filters_by_link() -> None:
    items = [make_result(0), make_result(1), make_result(2)]
    assert keep_new(items, {"https://example.com/1"}) == [items[0], items[2]]


def test_load_credentials_accepts_json_text_and_file_path(tmp_path: Path) -> None:
    key = {"type": "service_account", "client_email": "bot@example.iam.gserviceaccount.com"}
    key_file = tmp_path / "service_account.json"
    key_file.write_text(json.dumps(key), encoding="utf-8")

    assert load_credentials(json.dumps(key)) == key
    assert load_credentials(str(key_file)) == key


def test_open_worksheet_needs_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_CREDENTIALS", raising=False)
    with pytest.raises(MissingSettingError, match="GOOGLE_CREDENTIALS"):
        deliver.open_worksheet()


# ---- Telegram message --------------------------------------------------------------------


def test_format_message_lists_top_n_in_order() -> None:
    results = [make_result(i, relevance=5 - i) for i in range(4)]

    text = format_message(results, today=TODAY, top_n=2)

    assert text.startswith(f"AI Daily Digest - {TODAY}\nTop 2 of 4 new articles")
    assert "1. Title 0 [5/5]" in text
    assert "2. Title 1 [4/5]" in text
    assert "Title 2" not in text
    assert "https://example.com/0" in text


def test_format_message_without_results() -> None:
    assert "No new articles today." in format_message([], today=TODAY)


def test_format_message_never_exceeds_telegram_limit() -> None:
    long_results = [{**make_result(i), "summary": "x" * 1500} for i in range(5)]

    text = format_message(long_results, today=TODAY, top_n=5)  # type: ignore[arg-type]

    assert len(text) <= deliver.TELEGRAM_MAX_CHARS
    assert "1. Title 0" in text
    assert "5. Title 4" not in text  # whole articles are dropped, not cut in half


# ---- Telegram sending --------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code: int = 200, body: Any = None) -> None:
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = json.dumps(body) if body is not None else "<html>error</html>"
        self._body = body

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("not JSON")
        return self._body


def fake_post(monkeypatch: pytest.MonkeyPatch, response: FakeResponse) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return response

    monkeypatch.setattr(deliver.requests, "post", post)
    return calls


def test_send_telegram_posts_text_to_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_post(monkeypatch, FakeResponse(200, {"ok": True}))

    send_telegram("hello", token="123:ABC", chat_id="42")

    [call] = calls
    assert call["url"] == "https://api.telegram.org/bot123:ABC/sendMessage"
    assert call["json"]["chat_id"] == "42"
    assert call["json"]["text"] == "hello"
    assert call["timeout"] == 30


def test_send_telegram_reads_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_TOKEN", "999:XYZ")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "7")
    calls = fake_post(monkeypatch, FakeResponse(200, {"ok": True}))

    send_telegram("hi")

    assert "bot999:XYZ" in calls[0]["url"]
    assert calls[0]["json"]["chat_id"] == "7"


def test_send_telegram_shows_telegram_error_description(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_post(monkeypatch, FakeResponse(400, {"ok": False, "description": "chat not found"}))
    with pytest.raises(TelegramError, match="400: chat not found"):
        send_telegram("hello", token="123:ABC", chat_id="42")


def test_send_telegram_handles_non_json_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_post(monkeypatch, FakeResponse(502))
    with pytest.raises(TelegramError, match="502"):
        send_telegram("hello", token="123:ABC", chat_id="42")


def test_send_telegram_never_leaks_token_on_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def post(url: str, **kwargs: Any) -> None:
        raise requests.ConnectionError(f"Max retries exceeded with url: {url}")

    monkeypatch.setattr(deliver.requests, "post", post)

    with pytest.raises(TelegramError) as info:
        send_telegram("hello", token="123:SECRET", chat_id="42")
    assert "SECRET" not in str(info.value)
    assert info.value.__cause__ is None


def test_send_telegram_needs_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    with pytest.raises(MissingSettingError, match="TELEGRAM_TOKEN"):
        send_telegram("hello")
