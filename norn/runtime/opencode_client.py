"""OpenCode relay driver: chat completions through the OpenCode Zen/Go relay (NOR-30).

The relay is OpenAI-compatible for most models, but it adds three requirements
that :class:`~norn.runtime.openai_client.OpenAICompatibleClient` does not meet:

* ``x-opencode-session`` — an opaque, stable per-conversation value. Without it
  the relay answers ``400 MissingSessionID`` *before* routing the request, so
  the failure looks nothing like the real one.
* An identifying ``User-Agent``. A generic HTTP-library UA is rejected by
  Cloudflare with error ``1010``.
* Per-model transports. ``claude-*``, ``qwen*`` and (on Go) ``minimax-*`` are
  served through ``/v1/messages``; ``gpt-*``, ``grok-*`` and ``muse-spark``
  through ``/v1/responses``. Only the remaining models speak
  ``/v1/chat/completions``, which is the only transport Norn implements, so the
  driver rejects the rest with an explicit message instead of a bare 404.

Auth: the relay's own env var (``OPENCODE_GO_API_KEY`` / ``OPENCODE_ZEN_API_KEY``)
is resolved by :class:`~norn.domain.models.ModelConfig`; ``api_key`` in the
campaign YAML keeps working and wins over the environment.
"""
from __future__ import annotations

import hashlib

from norn.domain.models import ModelConfig, _opencode_family
from norn.runtime.openai_client import OpenAICompatibleClient
from norn.runtime.retry import DEFAULT_RETRY_ATTEMPTS

#: User agent sent on every relay request. Go's terms require a client that
#: identifies itself; a library default triggers Cloudflare 1010 or an
#: "unrecognized client" rejection.
RELAY_USER_AGENT = "norn/0.1.0 (llm-redteam)"

#: Session-affinity header the relay uses to pin a conversation to one backend
#: (which also keeps its prompt cache warm).
SESSION_HEADER = "x-opencode-session"

CHAT_COMPLETIONS = "chat_completions"
ANTHROPIC_MESSAGES = "anthropic_messages"
CODEX_RESPONSES = "codex_responses"

#: Prefixes served by ``/v1/responses`` on every relay family.
_RESPONSES_PREFIXES: tuple[str, ...] = ("gpt-", "grok-", "muse-spark")
#: Prefixes served by ``/v1/messages`` on every relay family.
_MESSAGES_PREFIXES: tuple[str, ...] = ("qwen",)
#: Extra ``/v1/messages`` prefixes per relay family.
_MESSAGES_BY_FAMILY = {
    "go": ("minimax-",),
    "zen": ("claude-",),
}

_DIAGNOSTICS = {
    "FreeUsageLimitError": (
        "the relay's anonymous free tier is exhausted for this IP; "
        "subscribe to OpenCode Go/Zen or retry later"
    ),
    "MissingSessionID": (
        "the relay needs the x-opencode-session header (this driver sets it; "
        "a hand-rolled HTTP call must send one too)"
    ),
    "Go requires a credential": (
        "the relay rejected the credential for this model; check the OpenCode "
        "plan and the workspace region (DeepSeek V4.1 Flash/Pro require the "
        "Global region in Settings > Privacy)"
    ),
}


def model_transport(provider: str, model_name: str) -> str:
    """Return the relay transport that serves ``model_name`` (NOR-30).

    Args:
        provider: Relay provider name (``opencode-go`` / ``opencode-zen``).
        model_name: Model id as accepted by the relay.

    Returns:
        One of ``chat_completions`` (supported by Norn), ``anthropic_messages``
        or ``codex_responses`` (both unsupported).
    """
    name = (model_name or "").lower()
    family = _opencode_family(provider) or "zen"
    if name.startswith(_RESPONSES_PREFIXES):
        return CODEX_RESPONSES
    messages_prefixes = _MESSAGES_PREFIXES + _MESSAGES_BY_FAMILY.get(family, ())
    if name.startswith(messages_prefixes):
        return ANTHROPIC_MESSAGES
    return CHAT_COMPLETIONS


def opencode_session_id(*parts: object) -> str:
    """Derive a stable, opaque relay session id from ``parts``.

    Stable means the same replica always presents the same value (so the relay
    can pin the backend and reuse its prompt cache); opaque means it carries no
    personal data. Empty parts are ignored, so callers can pass optional scope
    values without branching.
    """
    raw = "|".join(str(part) for part in parts if part not in (None, ""))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def diagnose_error(status: int, body: str) -> str | None:
    """Translate a relay error response into an actionable hint (NOR-30).

    Args:
        status: HTTP status code of the failed response.
        body: Response body snippet (the relay puts the useful part in JSON).

    Returns:
        A one-line hint, or ``None`` when the error needs no relay-specific
        explanation (the client's own message already covers it).
    """
    for needle, hint in _DIAGNOSTICS.items():
        if needle.lower() in body.lower():
            return hint
    if status == 403 and "1010" in body:
        return "Cloudflare rejected the request: send an identifying User-Agent"
    if status == 401:
        return "check api_key in the campaign config (or the relay's env var)"
    if status == 404:
        return "model not found on the relay: check model_name and base_url"
    return None


class OpenCodeRelayClient(OpenAICompatibleClient):
    """Chat-completions driver for the OpenCode Zen/Go relay.

    Only the ``chat_completions`` transport is implemented; models served
    through ``/v1/messages`` or ``/v1/responses`` raise a ``RuntimeError`` that
    names the transport instead of failing with a 404.
    """

    def __init__(
        self,
        *,
        session_id: str | None = None,
        retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    ):
        """Initialize the relay driver.

        Args:
            session_id: Session scope for every request of this client (for
                example ``opencode_session_id(campaign_id)``). When omitted it
                is derived from provider + base_url + model, which is stable
                per client and fine for a whole campaign.
            retry_attempts: Max HTTP attempts per request (1 disables retries).
        """
        super().__init__(retry_attempts=retry_attempts)
        self.session_id = session_id
        self._last_status = 0
        self._last_body = ""

    # ── Headers ──────────────────────────────────────────────────────────────

    def resolve_session(self, model_config: ModelConfig) -> str:
        """Return the session id for ``model_config``.

        Precedence: an explicit ``x-opencode-session`` in ``extra_headers``
        (caller wins), then the client-level ``session_id``, then a value
        derived from provider + base_url + model.
        """
        explicit = (model_config.extra_headers or {}).get(SESSION_HEADER)
        if explicit:
            return explicit
        if self.session_id:
            return self.session_id
        return opencode_session_id(
            model_config.provider, model_config.base_url, model_config.model_name
        )

    def relay_config(self, model_config: ModelConfig) -> ModelConfig:
        """Return a copy of ``model_config`` carrying the relay headers.

        Explicit ``extra_headers`` from the campaign override the defaults, so a
        user can pin a session or add a vendor tag without patching the driver.
        """
        headers = {
            "User-Agent": RELAY_USER_AGENT,
            SESSION_HEADER: self.resolve_session(model_config),
        }
        headers.update(model_config.extra_headers or {})
        return model_config.model_copy(update={"extra_headers": headers})

    # ── HTTP ─────────────────────────────────────────────────────────────────

    def _request(self, url: str, headers: dict, body: dict, timeout: float):
        """Remember the last response so failures can be diagnosed."""
        resp = super()._request(url, headers, body, timeout)
        self._last_status = resp.status_code
        try:
            self._last_body = resp.text[:1000]
        except Exception:  # noqa: BLE001 - a body we cannot read is not fatal
            self._last_body = ""
        return resp

    def _chat_request(self, model_config: ModelConfig, body: dict):
        """Guard the transport, add the relay headers, then delegate."""
        transport = model_transport(model_config.provider, model_config.model_name)
        if transport != CHAT_COMPLETIONS:
            raise RuntimeError(
                f"Model '{model_config.model_name}' is served by the OpenCode "
                f"relay through '{transport}', which Norn does not implement. "
                f"Use a chat_completions model (deepseek-*, glm-*, kimi-*) or "
                f"point the campaign at a different provider."
            )
        try:
            return super()._chat_request(self.relay_config(model_config), body)
        except RuntimeError as exc:
            hint = diagnose_error(self._last_status, self._last_body)
            if hint:
                raise RuntimeError(f"{exc} — relay: {hint}") from exc
            raise
