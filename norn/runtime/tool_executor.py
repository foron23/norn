"""Tool execution for the L3 agent loop (NOR-01, NOR-13).

The ``ToolExecutor`` holds a registry of lab tools (``{name, description,
input_schema, handler}``), exposes them as OpenAI/Ollama-compatible tool
schemas, and executes model tool calls with policy enforcement.

Every tool handler returns ``(result, authorized)``: the result text that is
injected back into the conversation and whether the call was authorized
(``is_authorized`` for metrics derives from this, not from the payload).
``execute()`` never raises for unknown tools or malformed arguments — it
returns an error result so the agent loop can keep going.

NOR-13 (declarative tools): a ``tools_file`` YAML can add extra tools with
generic handlers — ``mock`` (fixed result), ``http`` (POST JSON to a lab
URL) or ``subprocess`` (sandboxed command). Merge is ADD-ONLY: lab defaults
(``file_reader``, ``web_search``, ``send``) cannot be redefined. The
subprocess handler enforces a hard sandbox: command allowlist, mandatory
timeout and a fixed cwd inside ``sandbox/`` (fail-fast on anything else).
"""
from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import httpx
import pydantic
import yaml

Handler = Callable[[dict], tuple[str, bool]]

# Hard sandbox for the subprocess handler (NOR-13 D9): only these commands
# may be requested from a tools_file. Adding to this list is a code change.
SUBPROCESS_ALLOWLIST: frozenset[str] = frozenset({
    "ls", "cat", "echo", "pwd", "head", "tail", "grep", "wc", "date", "whoami",
})


class MockHandlerConfig(pydantic.BaseModel):
    type: Literal["mock"] = "mock"
    result: str = ""


class HttpHandlerConfig(pydantic.BaseModel):
    type: Literal["http"] = "http"
    url: str
    method: str = "POST"
    timeout: float = 5.0


class SubprocessHandlerConfig(pydantic.BaseModel):
    type: Literal["subprocess"] = "subprocess"
    command: list[str]
    timeout: float = pydantic.Field(ge=0.1, description="Mandatory timeout (seconds)")

    @pydantic.model_validator(mode="after")
    def _validate_command_allowlist(self) -> SubprocessHandlerConfig:
        if not self.command:
            raise ValueError("subprocess command must be a non-empty list")
        if self.command[0] not in SUBPROCESS_ALLOWLIST:
            raise ValueError(
                f"command {self.command[0]!r} not in subprocess allowlist: "
                f"{sorted(SUBPROCESS_ALLOWLIST)}"
            )
        return self


class DeclarativeToolConfig(pydantic.BaseModel):
    """One tool from a ``tools_file`` YAML (NOR-13)."""

    name: str
    description: str = ""
    input_schema: dict = pydantic.Field(default_factory=dict)
    handler: MockHandlerConfig | HttpHandlerConfig | SubprocessHandlerConfig
    authorized: bool | None = None  # None = handler decides; True/False = forced


def load_tools_file(path: str | Path) -> list[DeclarativeToolConfig]:
    """Load and validate a declarative tools YAML (fail-fast).

    Schema: a top-level ``tools:`` list. Unknown handler types, commands
    outside the subprocess allowlist, or malformed schemas raise a
    :class:`ValueError` with the file name — used by ``validate-config``
    and by the executor at construction time.
    """
    tools_path = Path(path)
    try:
        with open(tools_path) as f:
            data = yaml.safe_load(f)
    except OSError as exc:
        raise ValueError(f"Cannot read tools_file {tools_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in tools_file {tools_path}: {exc}") from exc

    if not isinstance(data, dict) or "tools" not in data:
        raise ValueError(
            f"tools_file {tools_path} must contain a top-level 'tools:' list"
        )
    try:
        return [DeclarativeToolConfig(**item) for item in data["tools"]]
    except pydantic.ValidationError as exc:
        raise ValueError(f"Invalid tool definition in {tools_path}: {exc}") from exc


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

    def __init__(self, tools: list[str] | None = None, sandbox_dir: str | Path | None = None,
                 tools_file: str | Path | None = None):
        self.sandbox_dir = Path(sandbox_dir) if sandbox_dir is not None else Path(self.DEFAULT_SANDBOX_DIR)
        self._registry: dict[str, ToolSpec] = {}
        self._register_defaults()
        if tools_file is not None:
            self._register_declarative(tools_file)
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

    # ── declarative tools (NOR-13) ───────────────────────────────────────

    def _register_declarative(self, tools_file: str | Path) -> None:
        """Register tools from a declarative YAML (ADD-ONLY merge, D10).

        Lab defaults (``file_reader``, ``web_search``, ``send``) cannot be
        redefined — a tools_file naming one of them is a hard error so the
        lab's policy can never be accidentally overridden.
        """
        for tool in load_tools_file(tools_file):
            if tool.name in self._registry:
                raise ValueError(
                    f"tools_file cannot redefine default tool '{tool.name}' "
                    f"(NOR-13 D10: tools_file only ADDS tools)"
                )
            self.register(
                tool.name,
                tool.description,
                tool.input_schema,
                self._wrap_authorized(self._handler_for(tool), tool.authorized),
            )

    def _handler_for(self, tool: DeclarativeToolConfig) -> Handler:
        """Build the generic handler for a declarative tool config."""
        cfg = tool.handler
        if cfg.type == "mock":
            return self._make_mock_handler(cfg)
        if cfg.type == "http":
            return self._make_http_handler(cfg)
        return self._make_subprocess_handler(cfg)

    @staticmethod
    def _make_mock_handler(cfg: MockHandlerConfig) -> Handler:
        """Fixed-result handler — safe for tests and lab mocks."""

        def handler(args: dict) -> tuple[str, bool]:
            return cfg.result, True

        return handler

    @staticmethod
    def _make_http_handler(cfg: HttpHandlerConfig) -> Handler:
        """POST the call arguments as JSON to the lab URL; the response
        body becomes the tool result. Non-2xx or network errors are
        unauthorized (the action did not complete)."""

        def handler(args: dict) -> tuple[str, bool]:
            try:
                resp = httpx.post(cfg.url, json=args, timeout=cfg.timeout)
                resp.raise_for_status()
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                return f"error: http request failed: {exc}", False
            return resp.text, True

        return handler

    def _make_subprocess_handler(self, cfg: SubprocessHandlerConfig) -> Handler:
        """Sandboxed subprocess handler (NOR-13 D9).

        The command is executed with a fixed cwd inside ``sandbox_dir`` and
        a mandatory timeout; the allowlist is validated at load time.
        Arguments are passed as a fixed argv list (no shell), so model
        input cannot smuggle shell metacharacters.
        """

        def handler(args: dict) -> tuple[str, bool]:
            argv = [str(a) for a in cfg.command]
            try:
                proc = subprocess.run(
                    argv,
                    capture_output=True,
                    text=True,
                    timeout=cfg.timeout,
                    cwd=str(self.sandbox_dir.resolve()),
                    check=False,  # returncode handled below (error result)
                )
            except (subprocess.TimeoutExpired, OSError) as exc:
                return f"error: subprocess failed: {exc}", False
            if proc.returncode != 0:
                return f"error: command exited {proc.returncode}: {proc.stderr.strip()}", False
            return proc.stdout.strip(), True

        return handler

    @staticmethod
    def _wrap_authorized(handler: Handler, authorized: bool | None) -> Handler:
        """Force the authorized flag when the YAML pins it (True/False);
        None (default) lets the handler decide (True unless error)."""

        if authorized is None:
            return handler

        def wrapped(args: dict) -> tuple[str, bool]:
            result, _ = handler(args)
            return result, authorized

        return wrapped

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
