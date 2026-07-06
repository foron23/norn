# Norn — LLM Red Teaming Framework

![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![Version](https://img.shields.io/badge/version-0.1.0-cyan)

A red teaming CLI framework for LLM applications — models, RAG systems, and agents.
Norn runs structured adversarial campaigns using a three-layer attack taxonomy and
produces scored reports with verified metrics, so security researchers can perform
reliable, reproducible LLM security audits.

> Point Norn at an LLM application, run a campaign, and trust the metrics in the report.

Norn was developed as part of the TFM (Master's thesis) *"Red teaming de aplicaciones LLM"*.
It is a research tool for the academic security evaluation of LLM applications.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Commands](#commands)
- [Configuration Reference](#configuration-reference)
- [Attack Taxonomy](#attack-taxonomy)
- [Example Configs](#example-configs)
- [Lab Experiments](#lab-experiments)
- [Project Structure](#project-structure)
- [Development](#development)
- [Constraints](#constraints)
- [License](#license)

## Features

- **Three-layer attack taxonomy** — 16 techniques across L1 (standalone LLM), L2 (RAG), and L3 (agents with tools), mapped to OWASP LLM Top 10 and MITRE ATLAS.
- **Pluggable providers** — talk to a local [Ollama](https://ollama.com) instance or any OpenAI-compatible endpoint (OpenAI API, Ollama `/v1`, vLLM, LM Studio, LocalAI, or a custom lab web app).
- **Pluggable scoring** — heuristic rules, simulated LLM judge, or hybrid mode with majority / weighted-avg / veto vote aggregation.
- **Reproducible metrics** — per-layer calculators (ASR, FAR/FRR, TTC, PSR@k, TDS, UAR, CTER, KCCR) stored with 95% confidence intervals.
- **Kill-chain analysis** — cross-layer KCCR (Kill-Chain Completion Rate) for end-to-end L1→L2→L3 compromise assessment.
- **Persisted runs** — every campaign, replica, turn, tool call, and scoring decision is stored in a SQLite (WAL, FK-enforced) database for full traceability.
- **Exportable reports** — JSON, CSV, and HTML reports via Jinja2 templates.
- **Black-box auditor** — Norn drives the target through its public API; the same campaign can be replayed against a baseline and a hardened target for A/B comparison.

## Installation

Requires **Python 3.11** or later.

```bash
git clone <repo-url> norn
cd norn

# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate         # Linux/macOS
# .venv\Scripts\activate          # Windows

# 2. Install Norn in editable mode
pip install -e .

# 3. (Optional) Install the development tooling
pip install -e ".[dev]"

# 4. Verify the installation
norn version
```

Dependencies: `typer`, `pydantic`, `pyyaml`, `jinja2`, `rich`, `tabulate` (see `pyproject.toml`).

### Provider prerequisites

Norn does not ship models. To run campaigns you need a reachable LLM endpoint:

- **Ollama** (default provider) — install from <https://ollama.com>, then pull a model:
  ```bash
  ollama pull llama3.1:8b
  ```
- **OpenAI-compatible** — set `provider: "openai"` in your campaign YAML and point `base_url` at your endpoint (e.g. `http://localhost:8085/v1/l1`). Provide `api_key` inline or via the `OLLAMA_API_KEY` environment variable.

## Quick Start

```bash
# 1. Initialize the SQLite database and seed the taxonomy catalog
norn init-db

# 2. Plan a campaign from a YAML config
norn plan-campaign -c examples/campaign_l1_baseline.yaml

# 3. Run the campaign (replace 1 with the campaign ID from step 2)
norn run-campaign --campaign-id 1

# 4. Export results (HTML, JSON, CSV)
norn export-campaign --campaign-id 1
```

Outputs are written to `./norn_exports/` by default. Open the HTML report in a browser
for a human-readable summary, or parse the JSON/CSV for further analysis.

### End-to-end example

```bash
norn init-db
norn validate-config examples/campaign_l2_rag.yaml          # dry-run validation
norn plan-campaign -c examples/campaign_l2_rag.yaml
norn list-campaigns                                         # find the campaign ID
norn run-campaign --campaign-id 2
norn show-campaign --campaign-id 2                          # metrics + aggregates
norn assess-campaign --campaign-id 2                        # re-run metric calc
norn export-campaign --campaign-id 2 -f html                # single-format export
```

## Commands

| Command | Description |
|---------|-------------|
| `norn init-db` | Initialize the SQLite database and seed the taxonomy catalog |
| `norn version` | Show Norn version and stack info |
| `norn validate-config <path>` | Validate a campaign YAML configuration file |
| `norn plan-campaign -c <path>` | Register a campaign in the DB and generate test cases |
| `norn run-campaign --campaign-id <id>` | Execute all test cases of a planned campaign |
| `norn list-campaigns` | List all campaigns with ID, name, layer, and state |
| `norn show-campaign --campaign-id <id>` | Show detailed campaign info including metric aggregates |
| `norn assess-campaign --campaign-id <id>` | Re-run metric calculations for an existing campaign |
| `norn compute-kccr --campaign-id <id>` | Compute Kill-Chain Completion Rate and risk assessment |
| `norn export-campaign --campaign-id <id>` | Export results to JSON, CSV, HTML, or all |
| `norn show-taxonomy` | Display the three-layer attack taxonomy and metric definitions |

Most commands accept a `--db <path>` option to use a database file other than the default `norn.db`.

## Configuration Reference

Campaigns are defined in YAML files. See `examples/` for complete configurations.

### Top-level fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `campaign_name` | string | — | Human-readable name for the campaign (required) |
| `layer` | string | — | Attack layer: `"L1"` (standalone), `"L2"` (RAG), `"L3"` (agents) (required) |
| `description` | string | `""` | Optional description of the campaign purpose |
| `replicas_per_case` | int | `5` | Number of repetitions per test case |
| `max_turns` | int | `10` | Maximum conversational turns per replica |
| `max_tool_calls` | int | `5` | Maximum tool invocations per turn (L3 only) |
| `techniques` | list | all | Technique IDs from the taxonomy to run |
| `metrics` | list | `[]` | Metric IDs to compute |

### Model configuration (`model`)

Norn supports two providers, selected with the `provider` field.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | string | `"ollama"` | `"ollama"` or `"openai"` (case-insensitive) |
| `scheme` | string | `"http"` | URL scheme (Ollama only) |
| `host` | string | `"localhost"` | Model server host (Ollama only) |
| `port` | int | `11434` | Model server port (Ollama only) |
| `base_url` | string | `"https://api.openai.com/v1"` | OpenAI-compatible base URL (OpenAI provider) |
| `api_key` | string \| null | env `OLLAMA_API_KEY` | Bearer token; falls back to the `OLLAMA_API_KEY` env var |
| `model_name` | string | `"llama3.1:8b"` | Model identifier as known to the provider |
| `temperature` | float | `0.0` | Sampling temperature (0.0 = deterministic) |
| `top_p` | float | `0.9` | Nucleus sampling threshold |
| `max_tokens` | int | `2048` | Maximum response tokens |
| `seed` | int \| null | `42` | RNG seed for reproducibility |
| `timeout` | float | `60.0` | Request timeout in seconds |

**Ollama example:**

```yaml
model:
  provider: "ollama"
  host: "localhost"
  port: 11434
  model_name: "llama3.1:8b"
  temperature: 0.0
  seed: 42
```

**OpenAI-compatible example:**

```yaml
model:
  provider: "openai"
  base_url: "http://localhost:8085/v1/l1"
  api_key: "not-needed"        # or set OLLAMA_API_KEY in your environment
  model_name: "qwen2.5:3b"
  temperature: 0.0
  timeout: 120.0
```

### Scoring configuration (`scoring`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | string | `"hybrid"` | `"heuristic"`, `"llm_judge"`, or `"hybrid"` |
| `vote_strategy` | string | `"majority"` | `"majority"`, `"weighted_avg"`, or `"veto"` |
| `heuristic_rules` | dict | `{}` | Custom regex rules for heuristic scoring |
| `judge_model` | string \| null | `null` | Model name for the LLM judge (simulated) |
| `acceptance_threshold` | float | `0.5` | Score threshold above which a response is accepted |

### Export configuration (`export`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_dir` | string | `"./norn_exports"` | Directory for exported files |
| `formats` | list | `["html", "json", "csv"]` | Output formats |

## Attack Taxonomy

Norn uses a three-layer taxonomy mapped to OWASP LLM Top 10 and MITRE ATLAS:

- **L1 — LLM Standalone** (10 techniques): direct prompt injection, roleplay jailbreaks, obfuscation, multi-turn escalation, system prompt leakage, training data extraction, adversarial prompts, safety probing, policy evasion.
- **L2 — RAG Systems** (3 techniques): indirect prompt injection, index poisoning, retrieval context manipulation.
- **L3 — Agents with Tools** (3 techniques): tool abuse, indirect injection in agentic flow, cross-tool prompt chaining.

Run `norn show-taxonomy` to see every technique and its metric definitions, or filter a
single layer with `norn show-taxonomy -l L1`.

### Probes

Default adversarial payloads live in `norn/corpus/{layer}/adversarial/probes.json`.
Each probe carries `variants` tagged with a `split` (`harmful`, `benign`, `borderline`)
which drives FAR/FRR computation. You can override the corpus by placing JSON files in a
`norn/corpus/{layer}/adversarial/` directory; otherwise the built-in fallback catalog is used.

## Example Configs

### General-purpose campaigns

| File | Layer | Provider | Model | Description |
|------|-------|----------|-------|-------------|
| `campaign_l1_baseline.yaml` | L1 | ollama | `llama3.1:8b` | Baseline L1 audit at temperature 0 |
| `campaign_l1_varied_temp.yaml` | L1 | ollama | — | L1 audit at temperature 0.7 for response-variation testing |
| `campaign_l2_rag.yaml` | L2 | ollama | `mistral:7b` | RAG poisoning audit |
| `campaign_l3_agent.yaml` | L3 | ollama | `qwen2.5:7b` | Agent tool-abuse audit |
| `campaign_all_layers.yaml` | L3 | ollama | `llama3.1:8b` | Full kill-chain audit covering all 16 techniques |

### Lab-integrated campaigns (OpenAI-compatible endpoint on `localhost:8085`)

| File | Layer | Model | Description |
|------|-------|-------|-------------|
| `campaign_l2_lab.yaml` | L2 | `qwen2.5:3b` | RAG audit aligned with the docker-compose lab (pgvector backend) |
| `campaign_l1_rag_app.yaml` | L1 | `qwen2.5:3b` | L1 audit via the lab web app's standalone endpoint |
| `campaign_l2_rag_app.yaml` | L2 | `qwen2.5:3b` | L2 audit via the lab web app's RAG endpoint |
| `campaign_l3_rag_app.yaml` | L3 | `qwen2.5:3b` | L3 audit via the lab web app's agent endpoint |
| `campaign_l1_gemma4_lab.yaml` | L1 | `gemma4:31b-cloud` | Fast L1 demo (~6 min, 3 techniques, 2 replicas) |

## Lab Experiments

The `examples/lab/` directory contains the full TFM experimental design: a model matrix of
4 local models × 3 layers, A/B hardening protocol, cloud-model configs, and runner scripts.
Norn acts as a black-box auditor — the *same* campaign YAML is replayed against a baseline
target and a hardened target, with the hardening toggle living on the target side.

- **Design & protocol:** [`examples/lab/README.md`](examples/lab/README.md)
- **Local-model campaigns:** `examples/lab/lab_l{1,2,3}_*.yaml`
- **Cloud-model campaigns:** `examples/lab/cloud/`
- **Runners:** `examples/lab/run_experiments.sh`, `examples/lab/cloud/run_cloud_experiments.sh`

Design specs for the tool-call pipeline and cloud backend live under [`docs/`](docs/).

## Project Structure

```
norn/
  cli/          Typer commands and CLI entry point
  domain/       Pydantic configs, dataclasses, enums, taxonomy catalog
  runtime/      Campaign lifecycle orchestrator + provider clients (Ollama, OpenAI-compat)
  scoring/      Pluggable scoring (heuristic, LLM judge, hybrid)
  metrics/      Per-layer metric calculators (ASR, FAR, PSR@k, KCCR, …)
  persistence/  SQLite schema (14 tables) and repository classes
  export/       JSON, CSV, HTML report exporters
  probes/       Built-in fallback adversarial payloads per layer
  corpus/       Default adversarial probes per layer (JSON)
  reports/      Jinja2 HTML report template
examples/       Campaign YAML configurations + lab experimental design
tests/          Pytest suite (metrics, providers, orchestrator, tool calls)
docs/           Design specs (tool calls, cloud backend)
scripts/        Helper scripts
```

## Development

```bash
# Install with dev tooling
pip install -e ".[dev]"

# Run the test suite (uses an in-memory SQLite DB; no model endpoint required)
pytest

# Lint
ruff check .
```

The test suite is self-contained: it builds an in-memory SQLite database (see
`tests/conftest.py`) and exercises the metric calculators, provider factory, OpenAI/Ollama
clients, and tool-call parsing without contacting any live model. Tests that require a
running endpoint are skipped automatically.

## Constraints

- **Providers:** Ollama and any OpenAI-compatible endpoint. The OpenAI-compatible client
  uses stdlib `urllib` only — no SDK dependencies.
- **Scoring:** Heuristic rules and hybrid scoring. The `llm_judge` path is simulated (no
  external judge model is called).
- **Runtime:** Synchronous Python. No async/await.
- **Storage:** SQLite with WAL mode and foreign-key enforcement.
- **Tech stack:** `typer`, `pydantic`, `pyyaml`, `jinja2`, `rich`, `tabulate` (see `pyproject.toml`).

## License

Licensed under the **Apache License, Version 2.0**. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

---

**Version:** 0.1.0 · **Python:** 3.11+ · **Author:** Iker Foronda
