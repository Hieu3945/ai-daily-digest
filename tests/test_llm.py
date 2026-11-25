"""Tests for src/llm.py. The Gemini client is replaced by a fake, so no network or key is needed."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src import llm


class FakeModels:
    """Records the arguments of generate_content and returns a canned reply."""

    def __init__(self, reply: str | None) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.reply)


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch) -> FakeModels:
    models = FakeModels(reply="hello")
    monkeypatch.setattr(llm, "get_client", lambda: SimpleNamespace(models=models))
    return models


def test_ask_llm_returns_text_and_uses_env_model(
    fake_models: FakeModels, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "test-model")

    assert llm.ask_llm("Hi") == "hello"
    call = fake_models.calls[0]
    assert call["model"] == "test-model"
    assert call["contents"] == "Hi"
    assert call["config"] is None


def test_ask_llm_json_mode_sets_mime_type(fake_models: FakeModels) -> None:
    llm.ask_llm("Return JSON", json_mode=True)

    config = fake_models.calls[0]["config"]
    assert config.response_mime_type == "application/json"


def test_ask_llm_returns_empty_string_when_model_returns_none(fake_models: FakeModels) -> None:
    fake_models.reply = None
    assert llm.ask_llm("Hi") == ""


def test_default_model_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert llm.get_model() == llm.DEFAULT_MODEL


def test_missing_api_key_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    llm.get_client.cache_clear()
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        llm.get_client()
