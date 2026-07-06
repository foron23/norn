"""Provider abstraction: Protocol contract and factory for LLM driver selection."""
from __future__ import annotations

from typing import Protocol

from norn.domain.models import ModelConfig
from norn.runtime.ollama_client import OllamaClient
from norn.runtime.openai_client import OpenAICompatibleClient


class ProviderProtocol(Protocol):
    """Contract for LLM provider chat drivers."""

    def chat(self, model_config: ModelConfig, prompt: str) -> tuple[str, int, int, float, list[dict] | None, dict | None]:
        """Send a single-turn chat request and return response + metadata.

        Returns:
            Tuple of (response_text, tokens_in, tokens_out, latency_ms, tool_calls, metadata).
            tool_calls is None or a list of dicts, each with keys:
              id, type, function.name, function.arguments, result, is_authorized, turn
            metadata is None or a dict with optional tfm_retrieval key.
        """
        ...


def build_provider(provider_name: str) -> OllamaClient | OpenAICompatibleClient:
    """Factory for provider client instances.

    Args:
        provider_name: "ollama" or "openai" (case-insensitive).

    Returns:
        A client object with a chat(model_config, prompt) method.

    Raises:
        ValueError: If provider_name is not recognized.
    """
    name = provider_name.lower()
    if name == "openai":
        return OpenAICompatibleClient()
    elif name == "ollama":
        return OllamaClient()
    else:
        raise ValueError(
            f"Unknown provider: '{provider_name}'. "
            f"Supported providers: ollama, openai"
        )
