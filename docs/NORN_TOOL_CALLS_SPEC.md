# Spec: Consume L3 Tool Calls in Norn

**Audience:** Norn maintainer (`~/Documents/master/Norn2`)
**Depends on:** Lab API exposing `tool_calls` in response (see `docs/LAB_TOOL_CALLS_SPEC.md`)
**Status:** Draft

---

## 1. Problem

Norn's `tool_call_event` SQLite table is always empty for L3 campaigns. The L3
metric calculators — `compute_uar`, `compute_cter`, `compute_kccr` in
`norn/metrics/l3_metrics.py` — all read from this table, so UAR = 0.0 and
CTER = 0.0 for every campaign. Only KCCR gets non-zero values for a few
campaigns because it uses `scoring_decision.acceptance_flag` instead of tool
calls.

The fix has two halves:
- **Lab side** (separate spec): return `tool_calls` in the API response
- **Norn side** (this spec): parse `tool_calls` from provider responses and
  store them in the database

---

## 2. Data flow today vs target

### Current

```
client.chat() → (text, tokens_in, tokens_out, latency_ms)
                    ↓
           only text is stored as turn_event
           tool_calls in response are IGNORED
                    ↓
           tool_call_event table = EMPTY
                    ↓
           UAR = 0, CTER = 0, KCCR ≈ 0
```

### Target

```
client.chat() → (text, tokens_in, tokens_out, latency_ms, tool_calls)
                    ↓                    ↓
           text → turn_event    tool_calls → repo.insert_tool_call()
                    ↓                    ↓
                              tool_call_event table = POPULATED
                                           ↓
                              UAR = real, CTER = real, KCCR = real
```

---

## 3. Files to change

### 3a. `norn/runtime/providers.py` — Update Protocol

**Line 14** — Change return type annotation:

```python
# Before
def chat(self, model_config: ModelConfig, prompt: str) -> tuple[str, int, int, float]:

# After
def chat(self, model_config: ModelConfig, prompt: str) -> tuple[str, int, int, float, list[dict] | None]:
```

Docstring updated:

```python
def chat(self, model_config: ModelConfig, prompt: str) -> tuple[str, int, int, float, list[dict] | None]:
    """Send a single-turn chat request and return response + metadata.

    Returns:
        Tuple of (response_text, tokens_in, tokens_out, latency_ms, tool_calls).
        tool_calls is None or a list of dicts, each with keys:
          id, type, function.name, function.arguments, result, is_authorized, turn
    """
    ...
```

---

### 3b. `norn/runtime/openai_client.py` — Parse tool_calls

**Line 121** — After extracting `response_text`, also extract `tool_calls`:

```python
# Before (line 121)
response_text = choices[0].get("message", {}).get("content", "")

# After
message = choices[0].get("message", {})
response_text = message.get("content", "")
tool_calls_raw = message.get("tool_calls", [])
```

**Line 126** — Return 5-tuple instead of 4-tuple:

```python
# Before
return response_text, tokens_in, tokens_out, latency_ms

# After
return response_text, tokens_in, tokens_out, latency_ms, tool_calls_raw if tool_calls_raw else None
```

**Line 20** — Update method signature:

```python
# Before
def chat(self, model_config: ModelConfig, prompt: str) -> tuple[str, int, int, float]:

# After
def chat(self, model_config: ModelConfig, prompt: str) -> tuple[str, int, int, float, list[dict] | None]:
```

**Line 39-50** — Optionally accept `tools` parameter for future standalone L3
runs (not required for the lab API path, but forward-looking):

```python
# In the body dict (line 42), conditionally add tools:
# if tools:
#     body["tools"] = tools
#     body["tool_choice"] = "auto"
```

Full method signature change:

```python
def chat(self, model_config: ModelConfig, prompt: str,
         tools: list[dict] | None = None) -> tuple[str, int, int, float, list[dict] | None]:
```

---

### 3c. `norn/runtime/ollama_client.py` — Parse tool_calls

Same pattern as openai_client.

**Line 120** — Extract tool_calls from response:

```python
# Before
response_text = parsed.get("message", {}).get("content", "")

# After
message = parsed.get("message", {})
response_text = message.get("content", "")
tool_calls_raw = message.get("tool_calls", [])
```

**Line 126** — Return 5-tuple:

```python
# Before
return response_text, tokens_in, tokens_out, latency_ms

# After
return response_text, tokens_in, tokens_out, latency_ms, tool_calls_raw if tool_calls_raw else None
```

**Line 24** — Update method signature (same as openai_client).

**Line 42-53** — Optionally accept `tools` parameter for the Ollama request body:

Ollama `/api/chat` supports `tools` since v0.3+:

```json
{
  "model": "...",
  "messages": [...],
  "stream": false,
  "tools": [
    {"type": "function", "function": {"name": "read_email", "description": "...", "parameters": {...}}}
  ]
}
```

```python
# In the body dict (line 42), conditionally add tools:
# if tools:
#     body["tools"] = tools
```

---

### 3d. `norn/runtime/campaign.py` — Store tool calls

This is the critical change. Three things to modify:

#### 3d.i. Unpack the 5-tuple (line 207)

```python
# Before
response, tokens_in, tokens_out, latency_ms = client.chat(model_config, case.payload)

# After
response, tokens_in, tokens_out, latency_ms, tool_calls = client.chat(model_config, case.payload)
```

#### 3d.ii. Store tool calls after the turn event (after line 217)

```python
repo.insert_turn_event(
    replica_id, 0, case.payload, response,
    tokens_in=tokens_in,
    tokens_out=tokens_out,
    latency_ms=latency_ms,
)

# NEW: store tool call events
if tool_calls and layer == "L3":
    for tc in tool_calls:
        func = tc.get("function", {})
        repo.insert_tool_call(
            replica_id=replica_id,
            tool_name=func.get("name", tc.get("name", "unknown")),
            tool_params=func.get("arguments", tc.get("arguments", "{}")),
            tool_result=tc.get("result", ""),
            is_authorized=bool(tc.get("is_authorized", 1)),
            turn=tc.get("turn", 0),
        )
```

#### 3d.iii. No changes to L1/L2 loops

L1 and L2 campaigns never have tool calls. The `layer == "L3"` guard prevents
unnecessary DB writes. No changes needed for the existing L1/L2 code paths.

---

### 3e. `norn/metrics/orchestrator.py` — No changes needed

The orchestrator at line 39 already fetches tool calls:

```python
tool_calls = self.campaign_repo.get_tool_calls(campaign_id)
```

And passes them to L3 calculators at lines 84, 87, 90:

```python
uar = compute_uar(tool_calls)
cter = compute_cter(tool_calls)
kccr = compute_kccr(observations, observations, tool_calls)
```

The calculators in `l3_metrics.py` already read `tc.get("is_authorized", 1)`,
`tc.get("tool_name", "")`, and `tc.get("replica_id", 0)` — all fields that
`CampaignRepository.get_tool_calls()` returns from the DB.

---

### 3f. `norn/persistence/database.py` — No changes needed

`insert_tool_call()` exists at line 388 and takes exactly the arguments used in
3d.ii. `get_tool_calls()` exists at line 416 and returns all columns (`SELECT
tce.*`).

---

## 4. Backward compatibility

| Scenario                                    | Impact                                    |
|---------------------------------------------|-------------------------------------------|
| Existing L1/L2 campaigns                    | Zero impact — `tool_calls` is None, guard prevents DB writes |
| L3 campaign against old lab (no tool_calls) | `tool_calls` is None, no tool calls stored — same as today (UAR=0, CTER=0) |
| L3 campaign against new lab (with tool_calls)| Tool calls stored, metrics become non-zero |
| Existing DB with old campaigns              | No schema change needed — table already exists |
| Client returns 4-tuple (future-proof)       | Python unpacking: `a,b,c,d,*rest = client.chat()` handles this; or wrap in adapter |

**Recommendation:** Wrap the unpack in `_safe_unpack()` or check `len()` before
unpacking, in case an old client still returns a 4-tuple:

```python
result = client.chat(model_config, case.payload)
response, tokens_in, tokens_out, latency_ms = result[:4]
tool_calls = result[4] if len(result) > 4 else None
```

This avoids breaking if someone upgrades the provider but not the campaign
runner, or vice versa.

---

## 5. Verification

### Pre-flight check

```bash
# Run a single L3 campaign
./run_experiments.sh --mode baseline --layers L3 --models gemma3_4b --quick --resume
```

### DB validation query

```sql
SELECT r.campaign_id, c.name, COUNT(t.id) as tool_calls,
       SUM(t.is_authorized) as authorized,
       COUNT(*) - SUM(t.is_authorized) as unauthorized
FROM run_replica r
JOIN tool_call_event t ON t.replica_id = r.id
JOIN campaign c ON c.id = r.campaign_id
WHERE c.layer = 'L3'
GROUP BY r.campaign_id;
```

### Metric validation

After running a full L3 campaign, the comparison report should show non-zero
UAR and CTER:

```
### L3 Metrics ###

--- UAR ---
MODEL           BASELINE     HARDENED     DELTA
llama3.1        0.3120       0.1846       -0.1274
...

--- CTER ---
MODEL           BASELINE     HARDENED     DELTA
llama3.1        0.4231       0.2769       -0.1462
...
```

---

## 6. Optional: standalone Ollama L3 support

Currently L3 YAML configs use `provider: openai` (→ lab API). For standalone
L3 audits calling Ollama directly, `plan_campaign()` at line 51-128 would need
to include tool definitions in test case metadata. The tools module at
`norn/probes/catalog.py` would need L3 tool definitions matching the lab's
`rag_app/app/tools.py`.

This is NOT required for the immediate fix. The lab API path is the primary
target.

---

## 7. Summary of files touched

| File                                   | Change                                               | Risk  |
|----------------------------------------|------------------------------------------------------|-------|
| `norn/runtime/providers.py`            | Update Protocol return type to 5-tuple               | LOW   |
| `norn/runtime/openai_client.py`        | Parse tool_calls, return 5-tuple                     | LOW   |
| `norn/runtime/ollama_client.py`        | Parse tool_calls, return 5-tuple                     | LOW   |
| `norn/runtime/campaign.py`             | Unpack 5-tuple, store tool calls via insert_tool_call | MED   |
| `norn/metrics/orchestrator.py`         | None (already reads tool_calls)                      | —     |
| `norn/persistence/database.py`         | None (methods already exist)                         | —     |
| `norn/metrics/l3_metrics.py`           | None (already handles populated data)                | —     |

---

*End of spec.*
