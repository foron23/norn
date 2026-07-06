"""Unit tests for build_provider() factory and ProviderProtocol."""
from __future__ import annotations

import pytest

from norn.runtime.providers import build_provider
from norn.runtime.ollama_client import OllamaClient
from norn.runtime.openai_client import OpenAICompatibleClient


def test_build_ollama():
    """build_provider('ollama') returns OllamaClient."""
    client = build_provider("ollama")
    assert isinstance(client, OllamaClient)


def test_build_openai():
    """build_provider('openai') returns OpenAICompatibleClient."""
    client = build_provider("openai")
    assert isinstance(client, OpenAICompatibleClient)


def test_build_openai_case_insensitive():
    """build_provider('OPENAI') (uppercase) returns OpenAICompatibleClient."""
    client = build_provider("OPENAI")
    assert isinstance(client, OpenAICompatibleClient)


def test_build_unknown_raises():
    """build_provider('unknown') raises ValueError."""
    with pytest.raises(ValueError, match="Unknown provider"):
        build_provider("unknown")
