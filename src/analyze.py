"""Analyze collected articles with the LLM: a short summary, a category and a relevance score.

Articles are sent in batches (one LLM call per batch, which is cheaper and faster than one
call per article). The LLM must answer with JSON, and every answer is validated before it
is used. A batch that still fails after a few retries is skipped, so one bad answer never
stops the daily run.

Run `python -m src.analyze` to analyze today's articles (needs GEMINI_API_KEY in .env).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, TypedDict

from src.collect import Article
from src.llm import MissingApiKeyError, ask_llm

logger = logging.getLogger(__name__)

BATCH_SIZE = 10
MAX_ATTEMPTS = 3  # first try + 2 retries
RETRY_DELAY_SECONDS = 2.0  # wait 2 s, then 4 s, ... between attempts

CATEGORIES = ["Product launch", "Research", "Business", "Policy", "Tutorial", "Other"]

PROMPT = """You are a news analyst for a company interested in: {topic}.

For EACH article below, return one JSON object with these keys:
- "id": the article id, copied exactly
- "summary": 2 short sentences in English, saying what happened and why it matters
- "category": one of {categories}
- "relevance": integer from 1 (not related to the topic) to 5 (must read)

Reply with a JSON list only, one object per article, no extra text.

Articles:
{articles}
"""


class AnalyzedArticle(TypedDict):
    """An article after analysis. `summary` is the LLM summary, not the feed text."""

    title: str
    link: str
    source: str
    published: str
    summary: str
    category: str
    relevance: int  # 1-5


class InvalidResponseError(ValueError):
    """The LLM answer is not the JSON we asked for."""


def build_prompt(batch: list[Article], topic: str) -> str:
    """Build the prompt for one batch.

    Each article gets a small numeric id. The LLM only has to copy the id back, and we take
    title and link from our own data, so the LLM can never change or invent a link.
    """
    items = [
        {"id": i, "title": a["title"], "source": a["source"], "text": a["summary"]}
        for i, a in enumerate(batch)
    ]
    return PROMPT.format(
        topic=topic,
        categories=", ".join(CATEGORIES),
        articles=json.dumps(items, ensure_ascii=False, indent=1),
    )


def _to_relevance(value: Any) -> int | None:
    """Turn 4, 4.0 or "4" into 4. Return None if it is not a whole number from 1 to 5."""
    if isinstance(value, bool):  # bool is a subclass of int in Python: reject True/False
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not number.is_integer() or not 1 <= number <= 5:
        return None
    return int(number)


def parse_response(raw: str, batch: list[Article]) -> list[AnalyzedArticle]:
    """Parse and validate the LLM answer for one batch.

    Raises InvalidResponseError if the answer is not usable at all (not JSON, not a list),
    so the caller can retry. Single bad items inside a good answer are just skipped.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidResponseError(f"Answer is not valid JSON: {exc}") from exc

    # Some models wrap the list in an object, e.g. {"articles": [...]},
    # or return a single object when the batch has only one article.
    if isinstance(data, dict):
        data = [data] if "id" in data else next(iter(data.values()), None)
    if not isinstance(data, list):
        raise InvalidResponseError(f"Expected a JSON list, got {type(data).__name__}.")

    results: dict[int, AnalyzedArticle] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        article_id = item.get("id")
        if isinstance(article_id, str) and article_id.isdigit():
            article_id = int(article_id)  # accept "3" as well as 3
        if not isinstance(article_id, int) or not 0 <= article_id < len(batch):
            logger.warning("Skipping item with unknown id: %r", article_id)
            continue
        summary = item.get("summary")
        relevance = _to_relevance(item.get("relevance"))
        if not isinstance(summary, str) or not summary.strip() or relevance is None:
            logger.warning("Skipping invalid item for id %d: %r", article_id, item)
            continue
        category = item.get("category")
        if category not in CATEGORIES:
            category = "Other"

        article = batch[article_id]
        results[article_id] = AnalyzedArticle(
            title=article["title"],
            link=article["link"],
            source=article["source"],
            published=article["published"],
            summary=summary.strip(),
            category=category,
            relevance=relevance,
        )

    if not results:
        raise InvalidResponseError("The answer contains no valid items.")
    if len(results) < len(batch):
        logger.warning("LLM analyzed only %d of %d articles", len(results), len(batch))
    return list(results.values())


def analyze_batch(batch: list[Article], topic: str) -> list[AnalyzedArticle]:
    """Analyze one batch, retrying on API errors or bad JSON. Returns [] if all attempts fail."""
    prompt = build_prompt(batch, topic)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return parse_response(ask_llm(prompt, json_mode=True), batch)
        except MissingApiKeyError:
            raise  # a setup problem: stop the run with a clear message instead of retrying
        except Exception as exc:  # API errors (quota, timeout, ...) or InvalidResponseError
            logger.warning("Batch attempt %d/%d failed: %s", attempt, MAX_ATTEMPTS, exc)
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_DELAY_SECONDS * attempt)

    logger.error("Skipping a batch of %d articles after %d attempts", len(batch), MAX_ATTEMPTS)
    return []


def analyze(
    articles: list[Article], topic: str, batch_size: int = BATCH_SIZE
) -> list[AnalyzedArticle]:
    """Analyze all articles in batches and return them sorted by relevance (highest first)."""
    results: list[AnalyzedArticle] = []
    for start in range(0, len(articles), batch_size):
        results.extend(analyze_batch(articles[start : start + batch_size], topic))
    # sorted() is stable: articles with the same score keep their original order
    return sorted(results, key=lambda r: r["relevance"], reverse=True)


if __name__ == "__main__":
    from src.collect import collect, load_config

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    collected = collect(config)
    analyzed = analyze(collected, topic=config.get("topic", "technology news"))
    print(f"Analyzed {len(analyzed)} of {len(collected)} articles")
    for item in analyzed[: int(config.get("top_n", 5))]:
        print(f"\n[{item['relevance']}/5] {item['title']} ({item['category']})")
        print(f"  {item['summary']}")
        print(f"  {item['link']}")
