"""Ollama HTTP client for chat completions via urllib.request.

Provides synchronous chat() calls to a running Ollama instance,
returning response text plus token counts and latency.
"""
from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request

from norn.domain.models import ModelConfig


class OllamaConnectionError(ConnectionError):
    """Raised when the Ollama server cannot be reached (connection refused)."""
    pass


class OllamaClient:
    """Synchronous HTTP client for Ollama /api/chat endpoint."""

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
        data = json.dumps(body).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if model_config.api_key:
            headers["Authorization"] = f"Bearer {model_config.api_key}"

        req = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="POST",
        )

        host = model_config.host
        port = model_config.port
        timeout = model_config.timeout

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raise RuntimeError(
                    "Ollama server returned 401 Unauthorized. "
                    "Check your api_key in the campaign config."
                ) from exc
            if exc.code == 404:
                try:
                    error_body = json.loads(exc.read())
                    server_msg = error_body.get("error", "")
                except Exception:
                    server_msg = ""
                msg = f"Model '{model_config.model_name}' not found on Ollama server at {host}:{port}"
                if server_msg:
                    msg += f": {server_msg}"
                raise RuntimeError(msg) from exc
            raise RuntimeError(
                f"Ollama server error (HTTP {exc.code}): {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, ConnectionRefusedError) or (
                hasattr(reason, 'errno') and reason.errno == 111
            ):
                raise OllamaConnectionError(
                    f"Cannot connect to Ollama at {host}:{port}. Is Ollama running?"
                ) from exc
            raise RuntimeError(
                f"Failed to connect to Ollama at {host}:{port}: {reason}"
            ) from exc
        except socket.timeout as exc:
            raise TimeoutError(
                f"Ollama request timed out after {timeout}s. "
                f"Check if the model is loaded or increase timeout in config."
            ) from exc
        except OSError as exc:
            if isinstance(exc, ConnectionRefusedError) or (
                hasattr(exc, 'errno') and exc.errno == 111
            ):
                raise OllamaConnectionError(
                    f"Cannot connect to Ollama at {host}:{port}. Is Ollama running?"
                ) from exc
            raise RuntimeError(
                f"Failed to connect to Ollama at {host}:{port}: {exc}"
            ) from exc

        try:
            parsed = json.loads(raw)
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
    def list_models(cls, host: str, port: int, timeout: float = 5.0, scheme: str = "http") -> list[str]:
        """Return list of available model names from the Ollama server."""
        import urllib.request
        import json
        url = f"{scheme}://{host}:{port}/api/tags"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return [m.get("name", "") for m in data.get("models", [])]
