"""Small fakes shared by the tests, so no test ever talks to Google or Telegram."""

from __future__ import annotations

from typing import Any

from src.analyze import AnalyzedArticle


class FakeWorksheet:
    """Keeps rows in a list and implements the gspread methods that src.deliver uses."""

    def __init__(self, rows: list[list[Any]] | None = None) -> None:
        self.rows: list[list[Any]] = [list(r) for r in rows or []]
        self.input_options: list[str] = []

    def row_values(self, row: int) -> list[Any]:
        return list(self.rows[row - 1]) if row <= len(self.rows) else []

    def col_values(self, col: int) -> list[Any]:
        return [r[col - 1] for r in self.rows if len(r) >= col]

    def append_row(self, values: list[Any], value_input_option: str = "RAW") -> None:
        self.append_rows([values], value_input_option)

    def append_rows(self, values: list[list[Any]], value_input_option: str = "RAW") -> None:
        self.input_options.append(value_input_option)
        self.rows.extend(list(v) for v in values)


def make_result(i: int, relevance: int = 3) -> AnalyzedArticle:
    return AnalyzedArticle(
        title=f"Title {i}",
        link=f"https://example.com/{i}",
        source="Sample",
        published="2026-11-27",
        summary=f"Summary {i}.",
        category="Research",
        relevance=relevance,
    )
