"""Small wrapper around the Gemini API (google-genai SDK).

Every other module talks to the LLM only through `ask_llm`, so switching to
another provider later means changing this one file.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

DEFAULT_MODEL = "gemini-3.1-flash-lite"


class MissingApiKeyError(RuntimeError):
    """GEMINI_API_KEY is not set. Retrying cannot fix this, so callers should not retry."""


def get_model() -> str:
    """Return the model name from GEMINI_MODEL (model names change often, so keep it in env)."""
    return os.getenv("GEMINI_MODEL") or DEFAULT_MODEL


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    """Create the Gemini client once and reuse it.

    The client is created lazily, so importing this module never needs an API key
    (useful for tests and for code paths that do not call the LLM).
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise MissingApiKeyError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return genai.Client(api_key=api_key)


def ask_llm(prompt: str, json_mode: bool = False) -> str:
    """Send a prompt to the LLM and return the text of its answer.

    Args:
        prompt: The full prompt text.
        json_mode: If True, ask the model to reply with valid JSON only
            (the caller is still responsible for parsing and validating it).
    """
    config = (
        types.GenerateContentConfig(response_mime_type="application/json") if json_mode else None
    )
    response = get_client().models.generate_content(
        model=get_model(), contents=prompt, config=config
    )
    return response.text or ""
