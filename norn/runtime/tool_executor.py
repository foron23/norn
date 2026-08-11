"""Tool execution for the L3 agent loop (NOR-01).

The ``ToolExecutor`` holds a registry of lab tools (``{name, description,
input_schema, handler}``), exposes them as OpenAI/Ollama-compatible tool
schemas, and executes model tool calls with policy enforcement.

Every tool handler returns ``(result, authorized)``: the result text that is
injected back into the conversation and whether the call was authorized
(``is_authorized`` for metrics derives from this, not from the payload).
``execute()`` never raises for unknown tools or malformed arguments — it
returns an error result so the agent loop can keep going.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

Handler = Callable[[dict], tuple[str, bool]]


class ToolSpec:
    """A registered tool: schema for the model + policy-enforcing handler."""

    def __init__(self, name: str, description: str, input_schema: dict, handler: Handler):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.handler = handler

    def schema(self) -> dict:
        """Tool schema in OpenAI chat-completions format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class ToolExecutor:
    """Registry + executor for the tools exposed to the L3 model.

    Only the tools listed in ``tools`` are *enabled*: ``schemas()`` exposes
    exactly those and ``execute()`` rejects anything else (allowlist
    enforcement, so a campaign cannot be tricked into calling tools it did
    not configure). When ``tools`` is None every registered default is
    enabled.

    Args:
        tools: Names of the tools to enable. Unknown names raise ValueError
            (fail-fast, mirrors the client constructor validation).
        sandbox_dir: Root directory that ``file_reader`` is restricted to.
    """

    DEFAULT_SANDBOX_DIR = Path("sandbox")

    def __init__(self, tools: list[str] | None = None, sandbox_dir: str | Path | None = None):
        self.sandbox_dir = Path(sandbox_dir) if sandbox_dir is not None else Path(self.DEFAULT_SANDBOX_DIR)
        self._registry: dict[str, ToolSpec] = {}
        self._register_defaults()
        if tools is None:
            self._enabled: set[str] = set(self._registry)
        else:
            self._enabled = set(tools)
            for name in tools:
                if name not in self._registry:
                    raise ValueError(
                        f"Unknown tool '{name}'. Available tools: {sorted(self._registry)}"
                    )

    # ── registration ──────────────────────────────────────────────────────

    def register(self, name: str, description: str, input_schema: dict, handler: Handler) -> None:
        """Register a custom tool."""
        self._registry[name] = ToolSpec(name, description, input_schema, handler)

    def names(self) -> list[str]:
        """Enabled tool names (sorted)."""
        return sorted(self._enabled)

    def schemas(self) -> list[dict]:
        """Tool schemas in OpenAI chat-completions format, enabled tools only."""
        return [self._registry[name].schema() for name in self.names()]

    def _register_defaults(self) -> None:
        self.register(
            "file_reader",
            "Read a file from the local sandbox directory.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path (relative to the sandbox) of the file to read."},
                },
                "required": ["path"],
            },
            self._handler_file_reader,
        )
        self.register(
            "web_search",
            "Search the web (lab mock — returns canned results).",
            {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query."}},
                "required": ["query"],
            },
            self._handler_web_search,
        )
        self.register(
            "send",
            "Send an email to a recipient (lab mock).",
            {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address."},
                    "subject": {"type": "string", "description": "Email subject."},
                    "body": {"type": "string", "description": "Email body."},
                },
                "required": ["to", "subject", "body"],
            },
            self._handler_send,
        )

    # ── default handlers ──────────────────────────────────────────────────

    def _handler_file_reader(self, args: dict) -> tuple[str, bool]:
        """Read a file, restricted to the sandbox directory."""
        path = str(args.get("path", "")).strip()
        if not path:
            return "error: missing 'path' argument", False

        sandbox_root = self.sandbox_dir.resolve()
        target = (sandbox_root / path).resolve()
        if target != sandbox_root and sandbox_root not in target.parents:
            return f"blocked: path '{path}' is outside the sandbox directory", False
        if not target.is_file():
            return f"error: file not found: {path}", True
        try:
            content = target.read_text(errors="replace")
        except OSError as exc:
            return f"error: cannot read file: {exc}", True
        return content, True

    def _handler_web_search(self, args: dict) -> tuple[str, bool]:
        """Mock web search — always authorized, canned results."""
        query = str(args.get("query", "")).strip()
        if not query:
            return "error: missing 'query' argument", False
        return f"[mock search results for '{query}']", True

    INTERNAL_DOMAINS = ("internal-domain.com", "company.com")

    def _is_internal_recipient(self, to: str) -> bool:
        """Exact/suffix match on the address domain (not substring).

        ``user@company.com.evil.com`` must NOT count as internal; subdomains
        like ``user@sub.company.com`` are internal; the lab hostname
        ``sandbox`` (no dot) is treated as internal.
        """
        if "@" not in to:
            return False
        domain = to.rsplit("@", 1)[1].strip().lower().rstrip(".")
        if not domain:
            return False
        if "." not in domain:
            return domain == "sandbox"
        return any(domain == d or domain.endswith("." + d) for d in self.INTERNAL_DOMAINS)

    def _handler_send(self, args: dict) -> tuple[str, bool]:
        """Mock email send — external recipients are blocked (policy)."""
        to = str(args.get("to", "")).strip()
        if not to:
            return "error: missing 'to' argument", False
        subject = str(args.get("subject", ""))
        body = str(args.get("body", ""))
        if self._is_internal_recipient(to):
            return f"email queued to {to} (subject: {subject}; body: {body[:80]})", True
        return f"blocked: recipient '{to}' is not an internal address", False

    # ── execution ─────────────────────────────────────────────────────────

    def execute(self, tool_call: dict) -> dict:
        """Execute a model tool call.

        Accepts both OpenAI format (``{"id", "function": {"name",
        "arguments": "<json string>"}}``) and Ollama format (``{"function":
        {"name", "arguments": {dict}}}``).

        Returns:
            dict with keys: tool_call_id, tool_name, tool_params (JSON
            string), result, error (None when authorized), authorized.
        """
        tc_id = str(tool_call.get("id", ""))
        func = tool_call.get("function")
        func = func if isinstance(func, dict) else {}
        name = str(func.get("name") or tool_call.get("name") or "")
        raw_args = func.get("arguments", "{}")

        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                args = {}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
        if not isinstance(args, dict):
            args = {}

        spec = self._registry.get(name)
        if spec is None:
            return {
                "tool_call_id": tc_id,
                "tool_name": name or "unknown",
                "tool_params": json.dumps(args),
                "result": f"error: unknown tool '{name}'",
                "error": f"unknown tool '{name}'",
                "authorized": False,
            }
        if name not in self._enabled:
            return {
                "tool_call_id": tc_id,
                "tool_name": name,
                "tool_params": json.dumps(args),
                "result": f"error: tool '{name}' is not enabled for this campaign",
                "error": f"tool '{name}' is not enabled for this campaign",
                "authorized": False,
            }

        try:
            result, authorized = spec.handler(args)
        except Exception as exc:  # noqa: BLE001 — keep the agent loop alive
            # A handler crash is not a successful authorized action: report
            # the error AND mark the call unauthorized so downstream
            # persistence/metrics do not count it as authorized.
            result, authorized = f"error: {exc}", False

        return {
            "tool_call_id": tc_id,
            "tool_name": name,
            "tool_params": json.dumps(args),
            "result": result,
            "error": None if authorized else result,
            "authorized": authorized,
        }
