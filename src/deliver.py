"""Deliver the digest: save every analyzed article to Google Sheets, send the top ones to Telegram.

Google Sheets is the long-term record (one row per article). It is also how we avoid
duplicates: a link that is already in the sheet is never analyzed or sent again.

Settings come from environment variables (see .env.example):
- GOOGLE_CREDENTIALS: the service account JSON, either pasted as one line or as a path to the file
- SHEET_ID: the long id in the sheet URL, between /d/ and /edit
- TELEGRAM_TOKEN and TELEGRAM_CHAT_ID: from @BotFather and getUpdates
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

import gspread
import requests

from src.analyze import AnalyzedArticle

logger = logging.getLogger(__name__)

HEADER = ["date", "title", "source", "category", "relevance", "summary", "link"]
LINK_COLUMN = HEADER.index("link") + 1  # gspread columns start at 1

TELEGRAM_URL = "https://api.telegram.org/bot{token}/sendMessage"
TELEGRAM_MAX_CHARS = 4096  # Telegram rejects longer messages


class MissingSettingError(RuntimeError):
    """A required environment variable is not set."""


class TelegramError(RuntimeError):
    """Telegram could not be reached or did not accept the message."""


class Worksheet(Protocol):
    """The few gspread worksheet methods we use. Tests pass a small fake instead."""

    def row_values(self, row: int) -> list[Any]: ...

    def col_values(self, col: int) -> list[Any]: ...

    def append_row(self, values: list[Any], value_input_option: str = ...) -> Any: ...

    def append_rows(self, values: list[list[Any]], value_input_option: str = ...) -> Any: ...


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise MissingSettingError(f"{name} is not set. Add it to .env or to the GitHub secrets.")
    return value


# ---- Google Sheets -----------------------------------------------------------------------


def load_credentials(value: str) -> dict[str, Any]:
    """Read the service account key from a JSON string or from a path to the JSON file.

    GitHub secrets hold the JSON text itself; locally it is easier to point to the file.
    """
    text = value.strip()
    if not text.startswith("{"):
        text = Path(text).read_text(encoding="utf-8")
    return json.loads(text)


def open_worksheet() -> Worksheet:
    """Open the first tab of the Google Sheet given by SHEET_ID."""
    credentials = load_credentials(_require_env("GOOGLE_CREDENTIALS"))
    client = gspread.service_account_from_dict(credentials)
    return client.open_by_key(_require_env("SHEET_ID")).sheet1


def ensure_header(ws: Worksheet) -> None:
    """Write the header row if the sheet is still empty."""
    if not ws.row_values(1):
        ws.append_row(HEADER, value_input_option="RAW")


def saved_links(ws: Worksheet) -> set[str]:
    """Return every link already stored in the sheet (the header row is skipped)."""
    return {str(link) for link in ws.col_values(LINK_COLUMN)[1:] if link}


def keep_new(items: Iterable[Any], seen_links: set[str]) -> list[Any]:
    """Keep only items (articles or analyzed articles) whose link is not in `seen_links`."""
    return [item for item in items if item["link"] not in seen_links]


def to_row(article: AnalyzedArticle, today: str) -> list[Any]:
    """Turn one analyzed article into a sheet row, in the same order as HEADER."""
    return [
        today,
        article["title"],
        article["source"],
        article["category"],
        article["relevance"],
        article["summary"],
        article["link"],
    ]


def save_to_sheet(ws: Worksheet, results: list[AnalyzedArticle], today: str) -> int:
    """Append new results to the sheet and return how many rows were added.

    Links already in the sheet are skipped, so running the pipeline twice a day is safe.
    "RAW" stores text exactly as given: a title starting with "=" stays text and is
    never run as a spreadsheet formula.
    """
    ensure_header(ws)
    new_results = keep_new(results, saved_links(ws))
    if new_results:
        ws.append_rows([to_row(r, today) for r in new_results], value_input_option="RAW")
    logger.info("Saved %d new rows to Google Sheets", len(new_results))
    return len(new_results)


# ---- Telegram ----------------------------------------------------------------------------


def format_message(results: list[AnalyzedArticle], today: str, top_n: int = 5) -> str:
    """Build the Telegram text for the `top_n` best articles (results are sorted already).

    Plain text is used on purpose: with Markdown, a "_" or "*" in a title would break the
    message. Articles that do not fit into Telegram's limit are left out, never cut in half.
    """
    if not results:
        return f"AI Daily Digest - {today}\n\nNo new articles today."

    top = results[:top_n]
    message = f"AI Daily Digest - {today}\nTop {len(top)} of {len(results)} new articles"
    for rank, r in enumerate(top, start=1):
        block = (
            f"\n\n{rank}. {r['title']} [{r['relevance']}/5]\n"
            f"{r['source']} | {r['category']}\n{r['summary']}\n{r['link']}"
        )
        if len(message) + len(block) > TELEGRAM_MAX_CHARS:
            logger.warning("Message too long: sending only the first %d articles", rank - 1)
            break
        message += block
    return message


def send_telegram(text: str, token: str | None = None, chat_id: str | None = None) -> None:
    """Send `text` to a Telegram chat. Raises if Telegram does not accept the message."""
    token = token or _require_env("TELEGRAM_TOKEN")
    chat_id = chat_id or _require_env("TELEGRAM_CHAT_ID")
    try:
        response = requests.post(
            TELEGRAM_URL.format(token=token),
            json={
                "chat_id": chat_id,
                "text": text[:TELEGRAM_MAX_CHARS],
                "link_preview_options": {"is_disabled": True},  # keep the digest compact
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        # The request URL contains the bot token, so never let it reach the logs.
        raise TelegramError(f"Could not reach Telegram ({type(exc).__name__})") from None

    if not response.ok:
        # Telegram explains the problem in "description", e.g. "Bad Request: chat not found"
        try:
            description = response.json().get("description", "")
        except ValueError:
            description = response.text[:200]
        raise TelegramError(f"Telegram error {response.status_code}: {description}")
    logger.info("Sent the digest to Telegram")
