"""Provider abstraction: Protocol contract and factory for LLM driver selection."""
from __future__ import annotations

from typing import Protocol

from norn.domain.models import ModelConfig, _opencode_family
from norn.runtime.ollama_client import OllamaClient
from norn.runtime.openai_client import OpenAICompatibleClient
from norn.runtime.opencode_client import OpenCodeRelayClient


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
        provider_name: "ollama", "openai", or an OpenCode relay name
            ("opencode-go" / "opencode-zen", case-insensitive). Bare
            "opencode" means Zen, matching the relay's default endpoint.

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
    elif _opencode_family(name):
        # NOR-30: the relay needs a session-affinity header and a per-model
        # transport guard, so it gets its own client instead of a base_url tweak.
        return OpenCodeRelayClient()
    else:
        raise ValueError(
            f"Unknown provider: '{provider_name}'. "
            f"Supported providers: ollama, openai, opencode-go, opencode-zen"
        )
