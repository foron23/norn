"""OpenAI-compatible HTTP client for chat completions via httpx (NOR-05).

Provides synchronous chat() calls to any OpenAI Chat Completions-compatible
endpoint (OpenAI API, Ollama /v1, vLLM, LM Studio, LocalAI, etc.), with
automatic retry/backoff for transient HTTP failures (429, 5xx).
"""
from __future__ import annotations

import json
import time

import httpx

from norn.domain.models import ModelConfig
from norn.runtime.retry import DEFAULT_RETRY_ATTEMPTS, with_retry


class OpenAICompatibleClient:
    """Synchronous HTTP client for OpenAI /v1/chat/completions endpoint."""

    def __init__(self, *, retry_attempts: int = DEFAULT_RETRY_ATTEMPTS):
        """Initialize the client.

        Args:
            retry_attempts: Max HTTP attempts per request (1 disables retries).

        Raises:
            ValueError: If ``retry_attempts`` is less than 1.
        """
        if retry_attempts < 1:
            raise ValueError("retry_attempts must be >= 1")
        self.retry_attempts = retry_attempts

    def _request(self, url: str, headers: dict, body: dict, timeout: float) -> httpx.Response:
        """POST JSON payload to ``url`` with retry/backoff.

        Returns the final :class:`httpx.Response` (after retries are
        exhausted for retryable status codes).
        """
        client = httpx.Client(timeout=timeout)
        try:
            return with_retry(
                lambda: client.post(url, headers=headers, json=body),
                attempts=self.retry_attempts,
            )
        finally:
            client.close()

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

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "norn/0.1.0",
        }
        if model_config.api_key:
            headers["Authorization"] = f"Bearer {model_config.api_key}"

        t0 = time.monotonic()

        # ── Error Handling ────────────────────────────────────────────────

        try:
            resp = self._request(url, headers, body, model_config.timeout)
        except httpx.TimeoutException as exc:
            raise TimeoutError(
                f"OpenAI-compatible request timed out after {model_config.timeout}s."
            ) from exc
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot connect to OpenAI-compatible endpoint "
                f"at {base_url}. Is the server running?"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Failed to connect to endpoint at {base_url}: {exc}"
            ) from exc

        latency_ms = (time.monotonic() - t0) * 1000.0

        if resp.status_code == 401:
            raise RuntimeError(
                "OpenAI-compatible endpoint returned 401 Unauthorized. "
                "Check your api_key in the campaign config."
            )
        if resp.status_code == 404:
            raise RuntimeError(
                f"Model '{model_config.model_name}' not found "
                f"at {base_url}. "
                "Check the model name or base_url."
            )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"OpenAI-compatible endpoint error "
                f"(HTTP {resp.status_code}): {resp.reason_phrase}"
            )

        # ── Response Parsing ──────────────────────────────────────────────

        try:
            parsed = resp.json()
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
                f"Response: {str(parsed)[:200]}"
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
