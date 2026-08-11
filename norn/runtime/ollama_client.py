"""Ollama HTTP client for chat completions via httpx (NOR-05).

Provides synchronous chat() calls to a running Ollama instance,
returning response text plus token counts and latency, with
retry/backoff for transient HTTP failures.
"""
from __future__ import annotations

import json

import httpx

from norn.domain.models import ModelConfig
from norn.runtime.retry import DEFAULT_RETRY_ATTEMPTS, with_retry


class OllamaConnectionError(ConnectionError):
    """Raised when the Ollama server cannot be reached (connection refused)."""


class OllamaClient:
    """Synchronous HTTP client for Ollama /api/chat endpoint."""

    def __init__(self, *, retry_attempts: int = DEFAULT_RETRY_ATTEMPTS):
        """Initialize the client.

        Args:
            retry_attempts: Max HTTP attempts per request (1 disables retries).
        """
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

    def chat(self, model_config: ModelConfig, prompt: str) -> tuple[str, int, int, float, list[dict] | None, None]:
        """Send a single-turn chat request to Ollama.

        Args:
            model_config: Model configuration with host, port, timeout,
                          and generation parameters.
            prompt: The user prompt string.

        Returns:
            Tuple of (response_text, tokens_in, tokens_out, latency_ms, tool_calls, metadata).
            Ollama never returns tfm_retrieval metadata, so the 6th element is always None.

        Raises:
            RuntimeError: On connection failure, HTTP error, or JSON parse error.
        """
        # Coerce seed: None → 0 (Ollama sentinel for "random seed")
        seed = model_config.seed if model_config.seed is not None else 0

        # max_tokens maps to Ollama's num_predict option
        body = {
            "model": model_config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "keep_alive": "5m",
            "options": {
                "temperature": model_config.temperature,
                "top_p": model_config.top_p,
                "seed": seed,
                "num_predict": model_config.max_tokens,
            },
        }

        url = f"{model_config.scheme}://{model_config.host}:{model_config.port}/api/chat"

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "norn/0.1.0",
        }
        if model_config.api_key:
            headers["Authorization"] = f"Bearer {model_config.api_key}"

        host = model_config.host
        port = model_config.port
        timeout = model_config.timeout

        try:
            resp = self._request(url, headers, body, timeout)
        except httpx.TimeoutException as exc:
            raise TimeoutError(
                f"Ollama request timed out after {timeout}s. "
                f"Check if the model is loaded or increase timeout in config."
            ) from exc
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {host}:{port}. Is Ollama running?"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Failed to connect to Ollama at {host}:{port}: {exc}"
            ) from exc

        if resp.status_code == 401:
            raise RuntimeError(
                "Ollama server returned 401 Unauthorized. "
                "Check your api_key in the campaign config."
            )
        if resp.status_code == 404:
            server_msg = ""
            try:
                server_msg = resp.json().get("error", "")
            except ValueError:
                pass
            msg = f"Model '{model_config.model_name}' not found on Ollama server at {host}:{port}"
            if server_msg:
                msg += f": {server_msg}"
            raise RuntimeError(msg)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Ollama server error (HTTP {resp.status_code}): {resp.reason_phrase}"
            )

        try:
            parsed = resp.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid response from Ollama at {host}:{port}"
            ) from exc

        message = parsed.get("message", {})
        response_text = message.get("content", "")
        tool_calls_raw = message.get("tool_calls", [])
        tokens_in = parsed.get("prompt_eval_count", 0)
        tokens_out = parsed.get("eval_count", 0)
        total_duration_ns = parsed.get("total_duration", 0)
        latency_ms = total_duration_ns / 1_000_000.0

        return response_text, tokens_in, tokens_out, latency_ms, tool_calls_raw if tool_calls_raw else None, None

    @classmethod
    def list_models(cls, host: str, port: int, timeout: float = 5.0, scheme: str = "http", api_key: str | None = None) -> list[str]:
        """Return list of available model names from the Ollama server."""
        url = f"{scheme}://{host}:{port}/api/tags"
        headers = {"User-Agent": "norn/0.1.0"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        client = httpx.Client(timeout=timeout)
        try:
            resp = client.get(url, headers=headers)
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Failed to list models from Ollama at {host}:{port} "
                    f"(HTTP {resp.status_code})"
                )
            return [m.get("name", "") for m in resp.json().get("models", [])]
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Failed to list models from Ollama at {host}:{port}: {exc}"
            ) from exc
        finally:
            client.close()
