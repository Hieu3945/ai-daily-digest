"""Run the whole pipeline once: collect -> skip known links -> analyze -> Google Sheets -> Telegram.

Usage:
    python main.py              # full run, needs every key in .env (or GitHub secrets)
    python main.py --dry-run    # collect + analyze and print the message; nothing is saved or sent

The exit code is 0 on success and 1 or 2 on failure, so GitHub Actions marks a broken run
as failed and emails the repo owner.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

from dotenv import load_dotenv

from src.analyze import analyze
from src.collect import DEFAULT_CONFIG_PATH, collect, load_config
from src.deliver import (
    MissingSettingError,
    format_message,
    keep_new,
    open_worksheet,
    save_to_sheet,
    saved_links,
    send_telegram,
)
from src.llm import MissingApiKeyError

logger = logging.getLogger("digest")


def run(config_path: str = str(DEFAULT_CONFIG_PATH), dry_run: bool = False) -> int:
    """Run the pipeline and return an exit code (0 = success)."""
    config = load_config(config_path)
    today = date.today().isoformat()

    articles = collect(config)
    if not articles:
        logger.error("No articles collected. Check the feed URLs in %s.", config_path)
        return 1

    worksheet = None
    if not dry_run:
        # Remove links we saved on earlier days BEFORE calling the LLM: old news is never
        # analyzed twice, which saves API quota and keeps the digest fresh.
        worksheet = open_worksheet()
        articles = keep_new(articles, saved_links(worksheet))
        logger.info("%d articles are new", len(articles))

    results = analyze(articles, topic=config.get("topic", "technology news")) if articles else []
    if articles and not results:
        logger.error("Every analysis batch failed, so nothing is saved or sent today.")
        return 1

    message = format_message(results, today=today, top_n=int(config.get("top_n", 5)))
    if dry_run:
        print(message)
        return 0

    # Save first: if Telegram fails, the articles are still recorded in the sheet.
    save_to_sheet(worksheet, results, today=today)
    send_telegram(message)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect, analyze and deliver the daily digest.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="path to config.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the Telegram message instead of saving to Sheets and sending it",
    )
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        return run(config_path=args.config, dry_run=args.dry_run)
    except (MissingApiKeyError, MissingSettingError) as exc:
        logger.error("Setup problem: %s", exc)  # a short message instead of a long traceback
        return 2


if __name__ == "__main__":
    sys.exit(main())
