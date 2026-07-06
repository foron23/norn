"""OpenAI-compatible HTTP client for chat completions via urllib.request.

Provides synchronous chat() calls to any OpenAI Chat Completions-compatible
endpoint (OpenAI API, Ollama /v1, vLLM, LM Studio, LocalAI, etc.).
"""
from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request

from norn.domain.models import ModelConfig


class OpenAICompatibleClient:
    """Synchronous HTTP client for OpenAI /v1/chat/completions endpoint."""

    def chat(self, model_config: ModelConfig, prompt: str) -> tuple[str, int, int, float, list[dict] | None, dict | None]:
        """Send a single-turn chat request to an OpenAI-compatible endpoint.

        Args:
            model_config: Model configuration with base_url, api_key, timeout,
                          and generation parameters.
            prompt: The user prompt string.

        Returns:
            Tuple of (response_text, tokens_in, tokens_out, latency_ms, tool_calls, metadata).

        Raises:
            RuntimeError: On HTTP error, JSON parse error, or empty response.
            ConnectionError: When the endpoint cannot be reached
                             (connection refused).
            TimeoutError: When the request exceeds the configured timeout.
        """
        # ── Request Construction ──────────────────────────────────────────

        body = {
            "model": model_config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": model_config.temperature,
            "top_p": model_config.top_p,
            "max_tokens": model_config.max_tokens,
        }
        # OpenAI API: seed must be omitted if None
        # (different from Ollama's 0 sentinel)
        if model_config.seed is not None:
            body["seed"] = model_config.seed

        base_url = model_config.base_url.rstrip("/")
        url = f"{base_url}/chat/completions"
        data = json.dumps(body).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "norn/0.1.0",
        }
        if model_config.api_key:
            headers["Authorization"] = f"Bearer {model_config.api_key}"

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        t0 = time.monotonic()

        # ── Error Handling ────────────────────────────────────────────────

        try:
            with urllib.request.urlopen(req, timeout=model_config.timeout) as resp:
                raw = resp.read().decode("utf-8")
                latency_ms = (time.monotonic() - t0) * 1000.0
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raise RuntimeError(
                    "OpenAI-compatible endpoint returned 401 Unauthorized. "
                    "Check your api_key in the campaign config."
                ) from exc
            elif exc.code == 404:
                raise RuntimeError(
                    f"Model '{model_config.model_name}' not found "
                    f"at {base_url}. "
                    "Check the model name or base_url."
                ) from exc
            raise RuntimeError(
                f"OpenAI-compatible endpoint error "
                f"(HTTP {exc.code}): {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, ConnectionRefusedError) or (
                hasattr(reason, "errno") and reason.errno == 111
            ):
                raise ConnectionError(
                    f"Cannot connect to OpenAI-compatible endpoint "
                    f"at {base_url}. Is the server running?"
                ) from exc
            raise RuntimeError(
                f"Failed to connect to endpoint at {base_url}: {reason}"
            ) from exc
        except socket.timeout as exc:
            raise TimeoutError(
                f"OpenAI-compatible request timed out after "
                f"{model_config.timeout}s."
            ) from exc

        # ── Response Parsing ──────────────────────────────────────────────

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid response from OpenAI-compatible endpoint "
                f"at {base_url}"
            ) from exc

        # Guard: empty choices list (some servers return this on error)
        choices = parsed.get("choices", [])
        if not choices:
            raise RuntimeError(
                f"OpenAI-compatible endpoint returned no choices. "
                f"Response: {raw[:200]}"
            )

        message = choices[0].get("message", {})
        response_text = message.get("content", "")
        tool_calls_raw = message.get("tool_calls", [])
        usage = parsed.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)

        metadata = {}
        tfm_retrieval = parsed.get("tfm_retrieval")
        if tfm_retrieval:
            metadata["tfm_retrieval"] = tfm_retrieval

        return response_text, tokens_in, tokens_out, latency_ms, tool_calls_raw if tool_calls_raw else None, metadata if metadata else None
