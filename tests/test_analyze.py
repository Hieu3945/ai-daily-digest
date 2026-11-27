"""Tests for src/analyze.py. The LLM is replaced by a fake, so no network or API key is needed."""

from __future__ import annotations

import json
from typing import Any

import pytest

from src import analyze as analyze_module
from src.analyze import InvalidResponseError, analyze, analyze_batch, build_prompt, parse_response
from src.collect import Article
from src.llm import MissingApiKeyError


def make_article(i: int) -> Article:
    return Article(
        title=f"Title {i}",
        link=f"https://example.com/{i}",
        summary=f"Feed text {i}",
        source="Sample",
        published="2026-11-27",
    )


def llm_item(i: int, relevance: Any = 3) -> dict[str, Any]:
    """One object as the LLM would return it for article id `i`."""
    return {"id": i, "summary": f"Summary {i}.", "category": "Research", "relevance": relevance}


class FakeLLM:
    """Returns the queued replies one by one. A queued Exception is raised instead."""

    def __init__(self, replies: list[str | Exception]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str, json_mode: bool = False) -> str:
        assert json_mode, "analyze must always ask for JSON"
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Skip the real waiting between retries and record the delays instead."""
    delays: list[float] = []
    monkeypatch.setattr(analyze_module.time, "sleep", delays.append)
    return delays


def use_fake_llm(monkeypatch: pytest.MonkeyPatch, replies: list[str | Exception]) -> FakeLLM:
    fake = FakeLLM(replies)
    monkeypatch.setattr(analyze_module, "ask_llm", fake)
    return fake


# ---- build_prompt -------------------------------------------------------------------------


def test_build_prompt_contains_topic_ids_and_titles_but_not_links() -> None:
    prompt = build_prompt([make_article(0), make_article(1)], topic="AI for shops")

    assert "AI for shops" in prompt
    assert '"id": 1' in prompt
    assert "Title 1" in prompt
    assert "https://example.com" not in prompt  # links come from our data, not from the LLM


# ---- parse_response -----------------------------------------------------------------------


def test_parse_response_merges_llm_fields_with_original_article() -> None:
    batch = [make_article(0), make_article(1)]
    raw = json.dumps([llm_item(1, relevance=5), llm_item(0, relevance=2)])

    results = parse_response(raw, batch)

    by_link = {r["link"]: r for r in results}
    first = by_link["https://example.com/0"]
    assert first["title"] == "Title 0"
    assert first["summary"] == "Summary 0."
    assert first["category"] == "Research"
    assert first["relevance"] == 2
    assert by_link["https://example.com/1"]["relevance"] == 5


def test_parse_response_accepts_list_wrapped_in_object_and_string_numbers() -> None:
    raw = json.dumps({"articles": [{**llm_item(0), "id": "0", "relevance": "4"}]})
    [result] = parse_response(raw, [make_article(0)])
    assert result["relevance"] == 4


def test_parse_response_accepts_single_object_for_one_article() -> None:
    [result] = parse_response(json.dumps(llm_item(0)), [make_article(0)])
    assert result["link"] == "https://example.com/0"


def test_parse_response_skips_invalid_items_and_keeps_valid_ones() -> None:
    batch = [make_article(i) for i in range(5)]
    raw = json.dumps(
        [
            llm_item(0),
            llm_item(1, relevance=9),  # out of range
            llm_item(2, relevance="high"),  # not a number
            {**llm_item(3), "summary": ""},  # empty summary
            llm_item(42),  # unknown id
            "not an object",
        ]
    )

    results = parse_response(raw, batch)

    assert [r["link"] for r in results] == ["https://example.com/0"]


def test_parse_response_replaces_unknown_category_with_other() -> None:
    raw = json.dumps([{**llm_item(0), "category": "Gossip"}])
    [result] = parse_response(raw, [make_article(0)])
    assert result["category"] == "Other"


@pytest.mark.parametrize(
    "raw",
    [
        "Sure! Here is the JSON you asked for",  # not JSON
        json.dumps("just a string"),  # JSON, but not a list
        json.dumps([llm_item(7)]),  # list without a single valid item
    ],
)
def test_parse_response_raises_for_unusable_answers(raw: str) -> None:
    with pytest.raises(InvalidResponseError):
        parse_response(raw, [make_article(0)])


# ---- analyze_batch (retries) --------------------------------------------------------------


def test_analyze_batch_retries_after_bad_json_then_succeeds(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    fake = use_fake_llm(monkeypatch, ["oops, not json", json.dumps([llm_item(0)])])

    results = analyze_batch([make_article(0)], topic="AI")

    assert len(results) == 1
    assert len(fake.prompts) == 2
    assert no_sleep == [analyze_module.RETRY_DELAY_SECONDS]


def test_analyze_batch_retries_after_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    use_fake_llm(monkeypatch, [RuntimeError("429 quota"), json.dumps([llm_item(0)])])
    assert len(analyze_batch([make_article(0)], topic="AI")) == 1


def test_analyze_batch_returns_empty_list_after_all_attempts_fail(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    fake = use_fake_llm(monkeypatch, ["bad"] * analyze_module.MAX_ATTEMPTS)

    assert analyze_batch([make_article(0)], topic="AI") == []
    assert len(fake.prompts) == analyze_module.MAX_ATTEMPTS
    assert len(no_sleep) == analyze_module.MAX_ATTEMPTS - 1  # no wait after the last attempt


def test_analyze_batch_does_not_retry_when_api_key_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = use_fake_llm(monkeypatch, [MissingApiKeyError("GEMINI_API_KEY is not set.")])

    with pytest.raises(MissingApiKeyError):
        analyze_batch([make_article(0)], topic="AI")
    assert len(fake.prompts) == 1


# ---- analyze ------------------------------------------------------------------------------


def test_analyze_sends_batches_and_sorts_by_relevance(monkeypatch: pytest.MonkeyPatch) -> None:
    articles = [make_article(i) for i in range(5)]
    fake = use_fake_llm(
        monkeypatch,
        [
            json.dumps([llm_item(0, 2), llm_item(1, 5)]),  # batch 1: articles 0-1
            json.dumps([llm_item(0, 1), llm_item(1, 4)]),  # batch 2: articles 2-3
            json.dumps([llm_item(0, 3)]),  # batch 3: article 4
        ],
    )

    results = analyze(articles, topic="AI", batch_size=2)

    assert len(fake.prompts) == 3
    assert [r["link"][-1] for r in results] == ["1", "3", "4", "0", "2"]
    assert [r["relevance"] for r in results] == [5, 4, 3, 2, 1]


def test_analyze_skips_a_failed_batch_and_keeps_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    articles = [make_article(i) for i in range(2)]
    use_fake_llm(
        monkeypatch,
        ["bad"] * analyze_module.MAX_ATTEMPTS + [json.dumps([llm_item(0, 4)])],
    )

    results = analyze(articles, topic="AI", batch_size=1)

    assert [r["link"] for r in results] == ["https://example.com/1"]


def test_analyze_with_no_articles_does_not_call_the_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = use_fake_llm(monkeypatch, [])
    assert analyze([], topic="AI") == []
    assert fake.prompts == []
