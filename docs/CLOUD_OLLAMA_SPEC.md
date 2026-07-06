# Cloud Ollama Backend — Implementation Spec

**For:** Lab developer  
**Status:** Draft  
**Date:** 2026-06-02

---

## 1. Current State

The lab already has stub env vars for cloud support (Phase 7.5), but they
only switch the **model name** — the base URL and client behavior remain
unchanged:

| File | What exists | What's missing |
|------|-------------|----------------|
| `config.py:20-23` | `OLLAMA_CLOUD_ENABLED`, `OLLAMA_CLOUD_API_KEY`, `OLLAMA_CLOUD_GENERATION_MODEL` | No `OLLAMA_CLOUD_BASE_URL` |
| `config.py:51-57` | `effective_generation_model` — returns cloud model name when enabled | No `effective_ollama_base_url` |
| `ollama_client.py:11` | `self._base_url = settings.ollama_base_url` (always local) | Cloud toggle has no effect on base URL |
| `docker-compose.yml:28` | `OLLAMA_API_KEY: ${OLLAMA_CLOUD_API_KEY:-}` passed to the Ollama *container* | No cloud base URL for the rag-app |
| `.env:40-47` | Cloud enabled, API key set, model `gemma4:31b-cloud` | — |

**Current flow (broken):**
```
Norn → lab:8085/v1/l2 → OllamaClient.generate() → http://ollama:11434/api/generate
                                                  → model: "gemma4:31b-cloud"
                                                  → Ollama container can't run it → 404/500
```

The Ollama container's `OLLAMA_API_KEY` feature (ollama.com cloud routing)
does not work reliably — the container can't transparently proxy cloud
models through `/api/generate`. The cloud API key should go directly into the
HTTP requests, not just the container env.

---

## 2. Required Changes

### 2.1 New env var: `OLLAMA_CLOUD_BASE_URL`

**File:** `config.py`, `docker-compose.yml`, `.env`

Add a separate base URL for cloud inference. When `OLLAMA_CLOUD_ENABLED=true`,
the lab switches to this endpoint instead of the local Ollama container.

```python
# config.py — add after line 23
ollama_cloud_base_url: str = Field(
    default="https://api.ollama.com",
    alias="OLLAMA_CLOUD_BASE_URL"
)
ollama_cloud_api_key: str = Field(default="", alias="OLLAMA_CLOUD_API_KEY")
ollama_cloud_generation_model: str = Field(default="", alias="OLLAMA_CLOUD_GENERATION_MODEL")
```

Add `effective_ollama_base_url` property:

```python
@property
def effective_ollama_base_url(self) -> str:
    if self.ollama_cloud_enabled:
        return self.ollama_cloud_base_url
    return self.ollama_base_url

@property
def effective_ollama_api_key(self) -> str | None:
    if self.ollama_cloud_enabled and self.ollama_cloud_api_key:
        return self.ollama_cloud_api_key
    return None
```

### 2.2 `OllamaClient` — use effective base URL + API key in requests

**File:** `rag_app/app/ollama_client.py`

**a)** Switch the base URL to respect the cloud toggle:

```python
# Change line 11 from:
self._base_url = settings.ollama_base_url.rstrip("/")
# To:
self._base_url = settings.effective_ollama_base_url.rstrip("/")
```

**b)** Inject `Authorization: Bearer <key>` header when cloud is enabled.
  Affects all three HTTP methods: `embed_texts()`, `generate()`,
  `classify_with_llama_guard()`, `list_models()`.

```python
def _headers(self) -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    api_key = settings.effective_ollama_api_key
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h
```

Then in each method, pass `headers=self._headers()` to `client.post()`.

**c)** Add trace logging at `DEBUG` level for cloud requests — log the URL,
  model, and status code so failures are diagnosable:

```python
logger.debug("Ollama request: %s %s model=%s", method, url, payload.get("model"))
```

### 2.3 Error handling — surface cloud-specific errors

**File:** `rag_app/app/ollama_client.py`, `rag_app/app/main.py`

The current error path wraps everything as HTTP 500 with a generic message.
When a cloud model is requested but the cloud endpoint is unreachable, the
lab should return a distinct error so Norn can report it clearly:

```python
# ollama_client.py
class OllamaCloudError(RuntimeError):
    """Cloud model unavailable or API key rejected."""
    pass

# In generate():
except httpx.HTTPStatusError as exc:
    if exc.response.status_code == 401:
        raise OllamaCloudError("Cloud API key rejected") from exc
    if exc.response.status_code == 404:
        raise OllamaCloudError(
            f"Model '{model_name}' not available in cloud"
        ) from exc
    raise
```

### 2.4 `docker-compose.yml` — add new env var

```yaml
# Under rag-app.environment, after line 68:
OLLAMA_CLOUD_BASE_URL: ${OLLAMA_CLOUD_BASE_URL:-https://api.ollama.com}
OLLAMA_CLOUD_ENABLED: ${OLLAMA_CLOUD_ENABLED:-false}
OLLAMA_CLOUD_API_KEY: ${OLLAMA_CLOUD_API_KEY:-}
OLLAMA_CLOUD_GENERATION_MODEL: ${OLLAMA_CLOUD_GENERATION_MODEL:-}
```

### 2.5 `.env` — update defaults

```env
# Ollama Cloud backend (Phase 7.5)
OLLAMA_CLOUD_ENABLED=true
OLLAMA_CLOUD_BASE_URL=https://api.ollama.com
OLLAMA_CLOUD_API_KEY=<your-key-from-ollama.com/settings/keys>
OLLAMA_CLOUD_GENERATION_MODEL=gemma4:31b-cloud
```

### 2.6 `agent_service.py` — update ChatOllama for L3

**File:** `rag_app/app/agent_service.py` (line ~32)

`ChatOllama` from langchain-ollama also needs the cloud base URL and API key:

```python
# Change:
llm_kwargs = {
    "model": model_name,
    "base_url": settings.ollama_base_url,
    ...
}
# To:
llm_kwargs = {
    "model": model_name,
    "base_url": settings.effective_ollama_base_url,
    ...
}

llm = ChatOllama(**llm_kwargs)

# If cloud API key is set, inject it into the ChatOllama client:
if settings.effective_ollama_api_key:
    llm.client.headers["Authorization"] = \
        f"Bearer {settings.effective_ollama_api_key}"
```

---

## 3. Norn Side — No Changes Needed

Norn's `ModelConfig` already supports `scheme` (http/https), `host`, and
`port` for direct Ollama. For lab experiments, Norn uses the `openai`
provider with the lab's `/v1/chat/completions` endpoint — no changes
required.

The Norn YAML config stays as-is:
```yaml
model:
  provider: "openai"
  base_url: "http://localhost:8085/v1/l2"
  model_name: "gemma4:31b-cloud"
```

---

## 4. Verification Checklist

- [ ] `OLLAMA_CLOUD_BASE_URL` added to `config.py`, `docker-compose.yml`, `.env`
- [ ] `effective_ollama_base_url` property in `config.py`
- [ ] `OllamaClient.__init__` uses `effective_ollama_base_url`
- [ ] `Authorization` header injected when cloud is enabled
- [ ] `agent_service.py` ChatOllama uses effective base URL + auth
- [ ] Cloud errors result in distinct error messages (not generic 500)
- [ ] `docker-compose up -d` with cloud enabled → lab app starts successfully
- [ ] Local models still work when `OLLAMA_CLOUD_ENABLED=false`
- [ ] Cloud model `gemma4:31b-cloud` returns valid responses
- [ ] `GET /v1/models` returns cloud model in listing when enabled

---

## 5. Files Affected

| File | Change |
|------|--------|
| `rag_app/app/config.py` | +2 fields, +2 properties |
| `rag_app/app/ollama_client.py` | Base URL switch, auth header, error handling |
| `rag_app/app/agent_service.py` | Base URL switch, auth header for ChatOllama |
| `docker-compose.yml` | +1 env var |
| `.env` | +1 field, doc update |
