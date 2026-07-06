# Norn — Framework de Red Teaming para LLMs

![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![Version](https://img.shields.io/badge/version-0.1.0-cyan)

**English** | [Español](README.es.md)

Un framework CLI de red teaming para aplicaciones basadas en LLMs — modelos, sistemas RAG y agentes.
Norn ejecuta campañas adversariales estructuradas usando una taxonomía de ataque de tres capas y
genera informes puntuados con métricas verificadas, de modo que los investigadores de seguridad
puedan realizar auditorías de seguridad sobre LLMs fiables y reproducibles.

> Apunta Norn a una aplicación LLM, ejecuta una campaña y confía en las métricas del informe.

---

## Tabla de Contenidos

- [Características](#características)
- [Instalación](#instalación)
- [Inicio Rápido](#inicio-rápido)
- [Comandos](#comandos)
- [Referencia de Configuración](#referencia-de-configuración)
- [Taxonomía de Ataques](#taxonomía-de-ataques)
- [Configuraciones de Ejemplo](#configuraciones-de-ejemplo)
- [Experimentos del Lab](#experimentos-del-lab)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Desarrollo](#desarrollo)
- [Restricciones](#restricciones)
- [Licencia](#licencia)

## Características

- **Taxonomía de ataque de tres capas** — 16 técnicas distribuidas en L1 (LLM standalone), L2 (RAG) y L3 (agentes con herramientas), mapeadas a OWASP LLM Top 10 y MITRE ATLAS.
- **Proveedores conectables** — comunica con una instancia local de [Ollama](https://ollama.com) o cualquier endpoint compatible con OpenAI (API de OpenAI, Ollama `/v1`, vLLM, LM Studio, LocalAI, o una web app de laboratorio propia).
- **Scoring conectable** — reglas heurísticas, juez LLM simulado, o modo híbrido con agregación de voto por mayoría / media ponderada / veto.
- **Métricas reproducibles** — calculadoras por capa (ASR, FAR/FRR, TTC, PSR@k, TDS, UAR, CTER, KCCR) almacenadas con intervalos de confianza del 95%.
- **Análisis de kill-chain** — KCCR (Kill-Chain Completion Rate) entre capas para evaluar compromisos extremo a extremo L1→L2→L3.
- **Ejecuciones persistentes** — cada campaña, réplica, turno, llamada a herramienta y decisión de scoring se almacena en una base de datos SQLite (modo WAL, con claves foráneas) para una trazabilidad completa.
- **Informes exportables** — informes en JSON, CSV y HTML mediante plantillas Jinja2.
- **Auditor black-box** — Norn conduce al objetivo a través de su API pública; la misma campaña puede reproducirse contra un objetivo baseline y un objetivo endurecido para comparación A/B.

## Instalación

Requiere **Python 3.11** o superior.

```bash
git clone <repo-url> norn
cd norn

# 1. Crear y activar un entorno virtual
python3 -m venv .venv
source .venv/bin/activate         # Linux/macOS
# .venv\Scripts\activate          # Windows

# 2. Instalar Norn en modo editable
pip install -e .

# 3. (Opcional) Instalar las herramientas de desarrollo
pip install -e ".[dev]"

# 4. Verificar la instalación
norn version
```

Dependencias: `typer`, `pydantic`, `pyyaml`, `jinja2`, `rich`, `tabulate` (ver `pyproject.toml`).

### Requisitos de los proveedores

Norn no distribuye modelos. Para ejecutar campañas necesitas un endpoint LLM alcanzable:

- **Ollama** (proveedor por defecto) — instala desde <https://ollama.com> y luego descarga un modelo:
  ```bash
  ollama pull llama3.1:8b
  ```
- **Compatible con OpenAI** — define `provider: "openai"` en tu YAML de campaña y apunta `base_url` a tu endpoint (p. ej. `http://localhost:8085/v1/l1`). Proporciona `api_key` en línea o mediante la variable de entorno `OLLAMA_API_KEY`.

## Inicio Rápido

```bash
# 1. Inicializar la base de datos SQLite y sembrar el catálogo de taxonomía
norn init-db

# 2. Planificar una campaña desde un YAML
norn plan-campaign -c examples/campaign_l1_baseline.yaml

# 3. Ejecutar la campaña (sustituye 1 por el ID de campaña del paso 2)
norn run-campaign --campaign-id 1

# 4. Exportar resultados (HTML, JSON, CSV)
norn export-campaign --campaign-id 1
```

Los resultados se escriben por defecto en `./norn_exports/`. Abre el informe HTML en un navegador
para un resumen legible, o procesa el JSON/CSV para análisis posteriores.

### Ejemplo de extremo a extremo

```bash
norn init-db
norn validate-config examples/campaign_l2_rag.yaml          # validación en seco
norn plan-campaign -c examples/campaign_l2_rag.yaml
norn list-campaigns                                         # localiza el ID de campaña
norn run-campaign --campaign-id 2
norn show-campaign --campaign-id 2                          # métricas + agregados
norn assess-campaign --campaign-id 2                        # recalcular métricas
norn export-campaign --campaign-id 2 -f html                # exportación de un único formato
```

## Comandos

| Comando | Descripción |
|---------|-------------|
| `norn init-db` | Inicializar la base de datos SQLite y sembrar el catálogo de taxonomía |
| `norn version` | Mostrar la versión de Norn e información del stack |
| `norn validate-config <path>` | Validar un archivo YAML de configuración de campaña |
| `norn plan-campaign -c <path>` | Registrar una campaña en la BD y generar los casos de prueba |
| `norn run-campaign --campaign-id <id>` | Ejecutar todos los casos de prueba de una campaña planificada |
| `norn list-campaigns` | Listar todas las campañas con ID, nombre, capa y estado |
| `norn show-campaign --campaign-id <id>` | Mostrar información detallada de la campaña incluyendo agregados de métricas |
| `norn assess-campaign --campaign-id <id>` | Recalcular las métricas de una campaña existente |
| `norn compute-kccr --campaign-id <id>` | Calcular el Kill-Chain Completion Rate y la evaluación de riesgo |
| `norn export-campaign --campaign-id <id>` | Exportar resultados a JSON, CSV, HTML o todos |
| `norn show-taxonomy` | Mostrar la taxonomía de ataques de tres capas y las definiciones de métricas |

La mayoría de comandos aceptan una opción `--db <path>` para usar un archivo de base de datos distinto del `norn.db` por defecto.

## Referencia de Configuración

Las campañas se definen en archivos YAML. Ver `examples/` para configuraciones completas.

### Campos de nivel superior

| Campo | Tipo | Por defecto | Descripción |
|-------|------|-------------|-------------|
| `campaign_name` | string | — | Nombre legible de la campaña (obligatorio) |
| `layer` | string | — | Capa de ataque: `"L1"` (standalone), `"L2"` (RAG), `"L3"` (agentes) (obligatorio) |
| `description` | string | `""` | Descripción opcional del propósito de la campaña |
| `replicas_per_case` | int | `5` | Número de repeticiones por caso de prueba |
| `max_turns` | int | `10` | Máximo de turnos conversacionales por réplica |
| `max_tool_calls` | int | `5` | Máximo de invocaciones de herramientas por turno (solo L3) |
| `techniques` | list | todas | IDs de técnicas de la taxonomía a ejecutar |
| `metrics` | list | `[]` | IDs de métricas a calcular |

### Configuración del modelo (`model`)

Norn soporta dos proveedores, seleccionados con el campo `provider`.

| Campo | Tipo | Por defecto | Descripción |
|-------|------|-------------|-------------|
| `provider` | string | `"ollama"` | `"ollama"` o `"openai"` (no distingue mayúsculas) |
| `scheme` | string | `"http"` | Esquema de URL (solo Ollama) |
| `host` | string | `"localhost"` | Host del servidor de modelos (solo Ollama) |
| `port` | int | `11434` | Puerto del servidor de modelos (solo Ollama) |
| `base_url` | string | `"https://api.openai.com/v1"` | URL base compatible con OpenAI (proveedor OpenAI) |
| `api_key` | string \| null | env `OLLAMA_API_KEY` | Bearer token; recurre a la variable de entorno `OLLAMA_API_KEY` |
| `model_name` | string | `"llama3.1:8b"` | Identificador del modelo según el proveedor |
| `temperature` | float | `0.0` | Temperatura de muestreo (0.0 = determinista) |
| `top_p` | float | `0.9` | Umbral de nucleus sampling |
| `max_tokens` | int | `2048` | Máximo de tokens de respuesta |
| `seed` | int \| null | `42` | Semilla RNG para reproducibilidad |
| `timeout` | float | `60.0` | Timeout de petición en segundos |

**Ejemplo con Ollama:**

```yaml
model:
  provider: "ollama"
  host: "localhost"
  port: 11434
  model_name: "llama3.1:8b"
  temperature: 0.0
  seed: 42
```

**Ejemplo compatible con OpenAI:**

```yaml
model:
  provider: "openai"
  base_url: "http://localhost:8085/v1/l1"
  api_key: "not-needed"        # o define OLLAMA_API_KEY en tu entorno
  model_name: "qwen2.5:3b"
  temperature: 0.0
  timeout: 120.0
```

### Configuración de scoring (`scoring`)

| Campo | Tipo | Por defecto | Descripción |
|-------|------|-------------|-------------|
| `mode` | string | `"hybrid"` | `"heuristic"`, `"llm_judge"` o `"hybrid"` |
| `vote_strategy` | string | `"majority"` | `"majority"`, `"weighted_avg"` o `"veto"` |
| `heuristic_rules` | dict | `{}` | Reglas regex personalizadas para el scoring heurístico |
| `judge_model` | string \| null | `null` | Nombre del modelo para el juez LLM (simulado) |
| `acceptance_threshold` | float | `0.5` | Umbral de puntuación por encima del cual se acepta una respuesta |

### Configuración de exportación (`export`)

| Campo | Tipo | Por defecto | Descripción |
|-------|------|-------------|-------------|
| `output_dir` | string | `"./norn_exports"` | Directorio para los archivos exportados |
| `formats` | list | `["html", "json", "csv"]` | Formatos de salida |

## Taxonomía de Ataques

Norn usa una taxonomía de tres capas mapeada a OWASP LLM Top 10 y MITRE ATLAS:

- **L1 — LLM Standalone** (10 técnicas): inyección de prompt directa, jailbreak por roleplay, ofuscación, escalado multi-turno, fuga de system prompt, extracción de datos de entrenamiento, prompts adversariales, sondeo de seguridad, evasión de políticas.
- **L2 — Sistemas RAG** (3 técnicas): inyección de prompt indirecta, envenenamiento de índice, manipulación del contexto de recuperación.
- **L3 — Agentes con Herramientas** (3 técnicas): abuso de herramientas, inyección indirecta en flujo agéntico, encadenamiento de prompts entre herramientas.

Ejecuta `norn show-taxonomy` para ver cada técnica y sus definiciones de métricas, o filtra una
sola capa con `norn show-taxonomy -l L1`.

### Probes

Los payloads adversariales por defecto viven en `norn/corpus/{layer}/adversarial/probes.json`.
Cada probe lleva `variants` etiquetadas con un `split` (`harmful`, `benign`, `borderline`)
que dirige el cálculo de FAR/FRR. Puedes sobrescribir el corpus colocando archivos JSON en un
directorio `norn/corpus/{layer}/adversarial/`; en caso contrario se usa el catálogo de respaldo integrado.

## Configuraciones de Ejemplo

### Campañas de propósito general

| Archivo | Capa | Proveedor | Modelo | Descripción |
|---------|------|-----------|--------|-------------|
| `campaign_l1_baseline.yaml` | L1 | ollama | `llama3.1:8b` | Auditoría L1 baseline a temperatura 0 |
| `campaign_l1_varied_temp.yaml` | L1 | ollama | — | Auditoría L1 a temperatura 0.7 para probar variación de respuestas |
| `campaign_l2_rag.yaml` | L2 | ollama | `mistral:7b` | Auditoría de envenenamiento RAG |
| `campaign_l3_agent.yaml` | L3 | ollama | `qwen2.5:7b` | Auditoría de abuso de herramientas en agente |
| `campaign_all_layers.yaml` | L3 | ollama | `llama3.1:8b` | Auditoría completa de kill-chain cubriendo las 16 técnicas |

### Campañas integradas con el lab (endpoint compatible con OpenAI en `localhost:8085`)

| Archivo | Capa | Modelo | Descripción |
|---------|------|--------|-------------|
| `campaign_l2_lab.yaml` | L2 | `qwen2.5:3b` | Auditoría RAG alineada con el lab docker-compose (backend pgvector) |
| `campaign_l1_rag_app.yaml` | L1 | `qwen2.5:3b` | Auditoría L1 vía el endpoint standalone de la web app del lab |
| `campaign_l2_rag_app.yaml` | L2 | `qwen2.5:3b` | Auditoría L2 vía el endpoint RAG de la web app del lab |
| `campaign_l3_rag_app.yaml` | L3 | `qwen2.5:3b` | Auditoría L3 vía el endpoint de agente de la web app del lab |
| `campaign_l1_gemma4_lab.yaml` | L1 | `gemma4:31b-cloud` | Demo rápida L1 (~6 min, 3 técnicas, 2 réplicas) |

## Experimentos del Lab

El directorio `examples/lab/` contiene el diseño experimental completo del TFM: una matriz de
modelos de 4 modelos locales × 3 capas, protocolo de endurecimiento A/B, configuraciones de
modelos en la nube y scripts de ejecución. Norn actúa como auditor black-box — el *mismo* YAML
de campaña se reproduce contra un objetivo baseline y un objetivo endurecido, con el conmutador
de endurecimiento viviendo en el lado del objetivo.

- **Diseño y protocolo:** [`examples/lab/README.md`](examples/lab/README.md)
- **Campañas de modelos locales:** `examples/lab/lab_l{1,2,3}_*.yaml`
- **Campañas de modelos en la nube:** `examples/lab/cloud/`
- **Runners:** `examples/lab/run_experiments.sh`, `examples/lab/cloud/run_cloud_experiments.sh`

Las especificaciones de diseño del pipeline de llamadas a herramientas y del backend en la nube
están en [`docs/`](docs/).

## Estructura del Proyecto

```
norn/
  cli/          Comandos Typer y entry point de la CLI
  domain/       Configs Pydantic, dataclasses, enums, catálogo de taxonomía
  runtime/      Orquestador del ciclo de vida de campaña + clientes proveedor (Ollama, OpenAI-compat)
  scoring/      Scoring conectable (heurístico, juez LLM, híbrido)
  metrics/      Calculadoras de métricas por capa (ASR, FAR, PSR@k, KCCR, …)
  persistence/  Esquema SQLite (14 tablas) y clases repositorio
  export/       Exportadores de informes JSON, CSV, HTML
  probes/       Payloads adversariales de respaldo por capa
  corpus/       Probes adversariales por defecto por capa (JSON)
  reports/      Plantilla Jinja2 de informe HTML
examples/       Configuraciones YAML de campaña + diseño experimental del lab
tests/          Suite pytest (métricas, proveedores, orquestador, llamadas a herramientas)
docs/           Especificaciones de diseño (llamadas a herramientas, backend en la nube)
scripts/        Scripts auxiliares
```

## Desarrollo

```bash
# Instalar con herramientas de desarrollo
pip install -e ".[dev]"

# Ejecutar la suite de tests (usa una BD SQLite en memoria; no requiere endpoint de modelo)
pytest

# Lint
ruff check .
```

La suite de tests es autocontenida: construye una base de datos SQLite en memoria (ver
`tests/conftest.py`) y ejercita las calculadoras de métricas, la fábrica de proveedores, los
clientes OpenAI/Ollama y el parseo de llamadas a herramientas sin contactar ningún modelo en
vivo. Los tests que requieren un endpoint en ejecución se omiten automáticamente.

## Restricciones

- **Proveedores:** Ollama y cualquier endpoint compatible con OpenAI. El cliente compatible con
  OpenAI usa solo `urllib` de la stdlib — sin dependencias de SDK.
- **Scoring:** Reglas heurísticas y scoring híbrido. La ruta `llm_judge` es simulada (no se
  llama a ningún modelo juez externo).
- **Runtime:** Python síncrono. Sin async/await.
- **Almacenamiento:** SQLite con modo WAL y aplicación de claves foráneas.
- **Stack tecnológico:** `typer`, `pydantic`, `pyyaml`, `jinja2`, `rich`, `tabulate` (ver `pyproject.toml`).

## Licencia

Licenciado bajo la **Apache License, Version 2.0**. Ver [`LICENSE`](LICENSE) y [`NOTICE`](NOTICE).

---

**Versión:** 0.1.0 · **Python:** 3.11+ · **Autor:** Iker Foronda
