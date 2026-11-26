# AI Daily Digest

Every morning, collect news from RSS feeds, let an LLM summarise and score each article,
save everything to Google Sheets and send the top stories to Telegram. Runs for free on
GitHub Actions.

> **Status:** work in progress. The full setup guide and demo are coming soon.

**Demo:** _screenshot of the Telegram message and Google Sheet coming soon_

## The problem

Small teams want to follow news about their industry and competitors, but nobody has time
to read dozens of sites every day. This pipeline reads them for you and sends a short,
ranked digest.

## How it works

1. **Collect** - read articles from RSS feeds listed in `config.yaml` (`feedparser`)
2. **Analyze** - send articles to Gemini in batches; get back JSON with a 2-sentence summary,
   a category and a relevance score (1-5)
3. **Store** - append the results to Google Sheets (`gspread`), skipping links already saved
4. **Deliver** - send the top 5 articles to a Telegram chat
5. **Schedule** - GitHub Actions runs the pipeline every day at 08:00 Vietnam time

## Project structure

```text
ai-daily-digest/
├── main.py                     # runs the whole pipeline (coming soon)
├── src/
│   ├── llm.py                  # ask_llm(): the only place that talks to Gemini
│   ├── collect.py              # read RSS/Atom feeds listed in config.yaml
│   ├── analyze.py              # LLM analysis as JSON (coming soon)
│   └── deliver.py              # Google Sheets + Telegram (coming soon)
├── tests/                      # pytest tests, no network or API key needed
├── config.yaml                 # feeds, topic and digest size - edit without touching code
├── .github/workflows/daily.yml # daily schedule (coming soon)
├── requirements.txt
└── .env.example
```

## Configuration

Everything a client usually wants to change lives in `config.yaml`:

```yaml
topic: "AI tools and automation for small businesses"
limit_per_feed: 10   # newest articles read from each feed
top_n: 5             # articles sent to Telegram
feeds:
  - name: Hacker News
    url: https://hnrss.org/frontpage
```

Check what would be collected today (no API key needed):

```bash
python -m src.collect
```

## Tech stack

Python 3.11+, Gemini API (`google-genai`), feedparser, gspread, Telegram Bot API,
GitHub Actions, pytest, ruff

## Run locally

```bash
git clone https://github.com/Hieu3945/ai-daily-digest.git
cd ai-daily-digest
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env             # then add your own keys
python -m pytest -q
```

## Limitations and next steps

_To be completed._
