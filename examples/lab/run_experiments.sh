#!/usr/bin/env bash
# ==============================================================================
# run_experiments.sh — Norn Lab Full Experiment Orchestrator
# ==============================================================================
# Executes the complete A/B hardening comparison for the TFM thesis:
#   4 models × 3 layers × 2 configs (baseline + hardened) = 24 campaign runs
#   Models: qwen3.5:2b, nemotron-3-nano:4b, qwen3.5:4b, gemma4:26b
#
# Usage:
#   ./run_experiments.sh                  # Run full experiment (A + B)
#   ./run_experiments.sh --mode baseline  # Only Config A
#   ./run_experiments.sh --mode hardened  # Only Config B
#   ./run_experiments.sh --check          # Validate prerequisites only
#   ./run_experiments.sh --quick          # Reduced replicas for smoke testing
#   ./run_experiments.sh --layers L1,L2   # Only specific layers
#   ./run_experiments.sh --models llama31,qwen25  # Only specific models
#   ./run_experiments.sh --resume         # Resume from last incomplete run
# ==============================================================================
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NORN_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LAB_DIR="${LAB_DIR:-${HOME}/Documents/master/TFM/lab}"
DB_FILE="${NORN_DIR}/norn_lab.db"
CAMPAIGNS_DIR="${SCRIPT_DIR}"
LOG_DIR="${NORN_DIR}/experiment_logs"
TRACK_FILE="${LOG_DIR}/campaign_tracker.csv"
SUMMARY_FILE="${LOG_DIR}/summary_$(date +%Y%m%d_%H%M%S).txt"

# Shell colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

# Models required in the Ollama container (order matters for pull time)
REQUIRED_MODELS=(
    "qwen3.5:2b"
    "nemotron-3-nano:4b"
    "qwen3.5:4b"
    "gemma4:26b"
    "nomic-embed-text"
)

# Campaign YAML files — all 12 combinations (4 models × 3 layers)
declare -A CAMPAIGN_FILES=(
    ["l1_qwen35_2b"]="lab_l1_qwen35_2b.yaml"
    ["l1_nemotron4b"]="lab_l1_nemotron4b.yaml"
    ["l1_qwen35_4b"]="lab_l1_qwen35_4b.yaml"
    ["l1_gemma4_26b"]="lab_l1_gemma4_26b.yaml"
    ["l2_qwen35_2b"]="lab_l2_qwen35_2b.yaml"
    ["l2_nemotron4b"]="lab_l2_nemotron4b.yaml"
    ["l2_qwen35_4b"]="lab_l2_qwen35_4b.yaml"
    ["l2_gemma4_26b"]="lab_l2_gemma4_26b.yaml"
    ["l3_qwen35_2b"]="lab_l3_qwen35_2b.yaml"
    ["l3_nemotron4b"]="lab_l3_nemotron4b.yaml"
    ["l3_qwen35_4b"]="lab_l3_qwen35_4b.yaml"
    ["l3_gemma4_26b"]="lab_l3_gemma4_26b.yaml"
)

# Defaults (overridable via CLI)
MODE="all"            # baseline | hardened | all
QUICK_MODE=false
CHECK_ONLY=false
RESUME=false
LAYERS=""             # comma-separated filter
MODELS=""             # comma-separated filter
DRY_RUN=false
REPORT_ONLY=false

# ── Help ───────────────────────────────────────────────────────────────────

usage() {
    cat << EOF
${BOLD}Norn Lab Experiment Orchestrator${NC}

Usage: $0 [OPTIONS]

${BOLD}Options:${NC}
  --mode baseline|hardened|all   Which hardening config to run (default: all)
  --layers L1,L2,L3              Filter by layers (default: all)
  --models llama31,mistral7b,... Filter by models (default: all)
  --check                        Validate prerequisites and exit
  --quick                        Reduced replicas (R=2) for smoke testing
  --dry-run                      Show what would run without executing
  --resume                       Skip already-completed campaigns
  --report-only                  Generate comparison report only (no campaigns)
  --db PATH                      Database path (default: norn_lab.db)
  --lab-dir PATH                 Path to TFM lab directory
  -h, --help                     Show this help

${BOLD}Examples:${NC}
  $0                                    # Full A/B experiment (24 runs)
  $0 --mode baseline --layers L1        # Only L1, baseline only
  $0 --quick --mode hardened            # Smoke-test hardened config only
  $0 --check                            # Validate setup without running
  $0 --resume                           # Resume interrupted experiment
  $0 --report-only                      # Regenerate comparison report from existing data
EOF
    exit 0
}

# ── Argument Parsing ───────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)       MODE="$2"; shift 2 ;;
        --layers)     LAYERS="$2"; shift 2 ;;
        --models)     MODELS="$2"; shift 2 ;;
        --check)      CHECK_ONLY=true; shift ;;
        --quick)      QUICK_MODE=true; shift ;;
        --dry-run)    DRY_RUN=true; shift ;;
        --resume)     RESUME=true; shift ;;
        --report-only) REPORT_ONLY=true; shift ;;
        --db)         DB_FILE="$2"; shift 2 ;;
        --lab-dir)    LAB_DIR="$2"; shift 2 ;;
        -h|--help)    usage ;;
        *)            echo "Unknown option: $1"; usage ;;
    esac
done

# ── Logging ────────────────────────────────────────────────────────────────

mkdir -p "${LOG_DIR}"

exec 3>&1 4>&2
exec 1> >(tee -a "${LOG_DIR}/experiment_$(date +%Y%m%d_%H%M%S).log") 2>&1

log()      { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $1"; }
success()  { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $1"; }
warn()     { echo -e "${YELLOW}[$(date +%H:%M:%S)] ⚠${NC} $1"; }
error()    { echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $1"; }
header()   { echo -e "\n${BOLD}━━━ $1 ━━━${NC}\n"; }
banner()   {
    echo -e "${CYAN}"
    echo "███╗   ██╗ ██████╗ ██████╗ ███╗   ██╗"
    echo "████╗  ██║██╔═══██╗██╔══██╗████╗  ██║"
    echo "██╔██╗ ██║██║   ██║██████╔╝██╔██╗ ██║"
    echo "██║╚██╗██║██║   ██║██╔══██╗██║╚██╗██║"
    echo "██║ ╚████║╚██████╔╝██║  ██║██║ ╚████║"
    echo "╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝"
    echo -e "${NC}  ${BOLD}LLM Red Teaming Framework — Lab Experiment Orchestrator${NC}"
    echo "  $(date)"
    echo ""
}

# ── Tracker CSV ────────────────────────────────────────────────────────────

init_tracker() {
    if [[ ! -f "${TRACK_FILE}" ]] || [[ "${RESUME}" == "false" ]]; then
        echo "run_key,config_mode,layer,model_name,campaign_id,state,started_at,finished_at,error_msg" > "${TRACK_FILE}"
    fi
}

track() {
    local run_key="$1" config_mode="$2" layer="$3" model="$4"
    local campaign_id="$5" state="$6" error_msg="${7:-}"
    local ts
    ts=$(date -Iseconds)
    # Update existing row or append new one
    if grep -q "^${run_key}," "${TRACK_FILE}" 2>/dev/null; then
        sed -i "s/^${run_key},.*/${run_key},${config_mode},${layer},${model},${campaign_id},${state},$(grep "^${run_key}," "${TRACK_FILE}" | cut -d, -f7),${ts},${error_msg}/" "${TRACK_FILE}"
    else
        echo "${run_key},${config_mode},${layer},${model},${campaign_id},${state},${ts},${ts}," >> "${TRACK_FILE}"
    fi
}

is_completed() {
    local run_key="$1"
    grep -q "^${run_key},.*,completed," "${TRACK_FILE}" 2>/dev/null
}

# ── Prerequisites ──────────────────────────────────────────────────────────

check_prerequisites() {
    header "Checking Prerequisites"

    local errors=0

    for cmd in docker curl sqlite3 python; do
        if command -v "$cmd" &>/dev/null; then
            success "$cmd found: $(command -v $cmd)"
        else
            error "$cmd not found"
            errors=$((errors + 1))
        fi
    done

    if python -c "import norn" 2>/dev/null; then
        success "norn Python package importable"
    else
        error "norn not installed (pip install -e .)"
        errors=$((errors + 1))
    fi

    if [[ -f "${LAB_DIR}/docker-compose.yml" ]]; then
        success "Lab docker-compose.yml found at ${LAB_DIR}"
    else
        error "Lab docker-compose.yml not found at ${LAB_DIR}"
        errors=$((errors + 1))
    fi

    if [[ -d "${CAMPAIGNS_DIR}" ]]; then
        local count
        count=$(ls "${CAMPAIGNS_DIR}"/lab_*.yaml 2>/dev/null | wc -l)
        success "${count} campaign YAML files found in ${CAMPAIGNS_DIR} (4 models × 3 layers)"
    else
        error "Campaign directory not found: ${CAMPAIGNS_DIR}"
        errors=$((errors + 1))
    fi

    if docker info &>/dev/null; then
        success "Docker daemon running"
    else
        error "Docker daemon not running"
        errors=$((errors + 1))
    fi

    if [[ $errors -gt 0 ]]; then
        error "${errors} prerequisite check(s) failed"
        return 1
    fi
    return 0
}

# ── Database ───────────────────────────────────────────────────────────────

init_database() {
    header "Initializing Database"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "[dry-run] Would run: norn init-db --db ${DB_FILE}"
        return 0
    fi

    python -m norn.cli.main init-db --db "${DB_FILE}" 2>&1 || {
        error "Database initialization failed"
        return 1
    }
    success "Database initialized: ${DB_FILE}"
    return 0
}

# ── Config Validation ──────────────────────────────────────────────────────

validate_all_configs() {
    header "Validating Campaign Configurations"

    local errors=0
    local campaigns=()
    mapfile -t campaigns < <(get_filtered_campaigns)

    if [[ ${#campaigns[@]} -eq 0 ]]; then
        warn "No campaigns match the current filters (layers=${LAYERS:-all}, models=${MODELS:-all})"
        return 1
    fi

    for key in "${campaigns[@]}"; do
        local yaml="${CAMPAIGN_FILES[$key]}"
        local path="${CAMPAIGNS_DIR}/${yaml}"

        log "Validating ${yaml}..."

        if [[ "$DRY_RUN" == "true" ]]; then
            log "  [dry-run] Would validate: ${path}"
            continue
        fi

        if python -m norn.cli.main validate-config "${path}" >/dev/null 2>&1; then
            success "  ${yaml}"
        else
            error "  ${yaml} — validation failed"
            errors=$((errors + 1))
        fi
    done

    if [[ $errors -gt 0 ]]; then
        error "${errors} config(s) failed validation"
        return 1
    fi
    return 0
}

# ── Filter Campaigns ───────────────────────────────────────────────────────

get_filtered_campaigns() {
    local keys=()
    for key in "${!CAMPAIGN_FILES[@]}"; do
        local layer model
        layer=$(echo "$key" | cut -d_ -f1 | tr '[:lower:]' '[:upper:]')  # l1→L1
        model=$(echo "$key" | cut -d_ -f2-)

        # Layer filter
        if [[ -n "${LAYERS}" ]]; then
            if ! echo ",${LAYERS}," | grep -qi ",${layer},"; then
                continue
            fi
        fi

        # Model filter
        if [[ -n "${MODELS}" ]]; then
            if ! echo ",${MODELS}," | grep -qi ",${model},"; then
                continue
            fi
        fi

        keys+=("$key")
    done
    printf '%s\n' "${keys[@]}"
}

# ── Lab Lifecycle ──────────────────────────────────────────────────────────

start_lab() {
    local config_mode="$1"  # "baseline" or "hardened"

    header "Starting Lab — Config: ${config_mode}"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "[dry-run] Would start lab with ${config_mode} hardening"
        return 0
    fi

    cd "${LAB_DIR}"

    # Stop any running containers first
    docker compose down --remove-orphans 2>/dev/null || true
    sleep 2

    # Set hardening environment variables
    local hardening_flag="false"
    if [[ "$config_mode" == "hardened" ]]; then
        hardening_flag="true"
    fi

    log "Starting services (L1_HARDENING=${hardening_flag}, L2_HARDENING=${hardening_flag}, L3_HARDENING=${hardening_flag})..."

    L1_HARDENING="${hardening_flag}" \
    L2_HARDENING="${hardening_flag}" \
    L3_HARDENING="${hardening_flag}" \
    AUTO_INGEST_ON_STARTUP="true" \
        docker compose up -d --wait 2>&1 || {
        error "docker compose up failed"
        return 1
    }

    # Wait for health
    wait_for_healthy

    # Restore working directory after docker compose (start_lab cd's into LAB_DIR)
    cd "${NORN_DIR}"
}

wait_for_healthy() {
    log "Waiting for services to become healthy..."

    cd "${LAB_DIR}"

    # Wait for app health endpoint (max 120s)
    for i in $(seq 1 60); do
        if curl -sf http://localhost:8085/health >/dev/null 2>&1; then
            success "Lab API healthy at http://localhost:8085"
            break
        fi
        if [[ $i -eq 60 ]]; then
            error "Lab API did not become healthy within 120s"
            docker compose logs --tail 50 rag-app 2>&1
            return 1
        fi
        sleep 2
    done

    # Show hardening status
    local health_json
    health_json=$(curl -sf http://localhost:8085/health 2>/dev/null || echo "{}")
    local l1 l2 l3
    l1=$(echo "$health_json" | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('l1_hardening','unknown'))" 2>/dev/null || echo "unknown")
    l2=$(echo "$health_json" | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('l2_hardening','unknown'))" 2>/dev/null || echo "unknown")
    l3=$(echo "$health_json" | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('l3_hardening','unknown'))" 2>/dev/null || echo "unknown")
    success "Hardening state — L1: ${l1}, L2: ${l2}, L3: ${l3}"
}

stop_lab() {
    header "Stopping Lab"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "[dry-run] Would stop lab"
        return 0
    fi

    cd "${LAB_DIR}"
    docker compose down --remove-orphans 2>/dev/null || true
    sleep 3
    success "Lab stopped"
}

ensure_models() {
    header "Ensuring Ollama Models"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "[dry-run] Would verify models: ${REQUIRED_MODELS[*]}"
        return 0
    fi

    # Check if Ollama container is running
    if ! docker ps --format '{{.Names}}' | grep -q 'tfm-rag-ollama'; then
        warn "Ollama container not running — skipping model check (start lab first)"
        return 0
    fi

    local available
    available=$(docker exec tfm-rag-ollama ollama list 2>/dev/null || echo "")

    for model in "${REQUIRED_MODELS[@]}"; do
        local model_base="${model%%:*}"
        if echo "$available" | grep -q "^${model_base}"; then
            success "Model available: ${model}"
        else
            log "Pulling model: ${model} (this may take several minutes)..."
            if docker exec tfm-rag-ollama ollama pull "${model}" >/dev/null 2>&1; then
                success "  Pulled: ${model}"
            else
                warn "  Failed to pull: ${model} — campaigns using this model may fail"
            fi
        fi
    done
}

# ── Campaign Execution ─────────────────────────────────────────────────────

plan_campaign() {
    local yaml_path="$1"
    local config_mode="$2"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "  [dry-run] Would plan: ${yaml_path}"
        echo "0"  # dummy ID
        return 0
    fi

    local effective_yaml="${yaml_path}"

    # If quick mode, create a temp YAML with reduced replicas
    if [[ "$QUICK_MODE" == "true" ]]; then
        effective_yaml=$(mktemp --suffix=.yaml)
        python -c "
import yaml
with open('${yaml_path}') as f:
    config = yaml.safe_load(f)
config['replicas_per_case'] = 2
with open('${effective_yaml}', 'w') as f:
    yaml.dump(config, f)
"
        log "  Quick mode: replicas reduced to 2 (temp: ${effective_yaml})" >&2
    fi

    # Run plan-campaign — display goes to stderr, only ID goes to stdout
    python -m norn.cli.main plan-campaign -c "${effective_yaml}" --db "${DB_FILE}" >&2
    local rc=$?

    # Cleanup temp YAML if created
    if [[ "$effective_yaml" != "${yaml_path}" ]]; then
        rm -f "${effective_yaml}"
    fi

    if [[ $rc -ne 0 ]]; then
        echo "0"
        return 1
    fi

    # Get campaign ID from DB (reliable, no ANSI/Rich parsing issues)
    local campaign_id
    campaign_id=$(sqlite3 "${DB_FILE}" "SELECT MAX(id) FROM campaign;" 2>/dev/null || echo "0")

    echo "$campaign_id"
}

run_campaign() {
    local campaign_id="$1"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "  [dry-run] Would run campaign ID: ${campaign_id}"
        return 0
    fi

    log "  Running campaign ${campaign_id}..."
    python -m norn.cli.main run-campaign -id "${campaign_id}" --db "${DB_FILE}" 2>&1
    local rc=$?

    if [[ $rc -eq 0 ]]; then
        success "  Campaign ${campaign_id} completed"
    else
        warn "  Campaign ${campaign_id} completed with warnings (rc=${rc})"
    fi

    return $rc
}

export_campaign() {
    local campaign_id="$1"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "  [dry-run] Would export campaign ID: ${campaign_id}"
        return 0
    fi

    log "  Exporting campaign ${campaign_id}..."
    python -m norn.cli.main export-campaign -id "${campaign_id}" -f all --db "${DB_FILE}" 2>&1 || true
    success "  Campaign ${campaign_id} exported"
}

execute_campaign() {
    local key="$1" config_mode="$2"
    local yaml="${CAMPAIGN_FILES[$key]}"
    local yaml_path="${CAMPAIGNS_DIR}/${yaml}"
    local layer model
    layer=$(echo "$key" | cut -d_ -f1 | tr '[:lower:]' '[:upper:]')
    model=$(echo "$key" | cut -d_ -f2-)
    local run_key="${config_mode}_${key}"

    header "Campaign: ${config_mode^^} / ${layer} / ${model}"
    log "YAML: ${yaml_path}"

    # Skip if already completed (resume mode)
    if [[ "$RESUME" == "true" ]] && is_completed "$run_key"; then
        success "Already completed — skipping (use --resume=false to re-run)"
        return 0
    fi

    track "$run_key" "$config_mode" "$layer" "$model" "" "planning" ""

    # Phase 1: Plan
    log "Planning campaign..."
    local campaign_id
    campaign_id=$(plan_campaign "${yaml_path}" "${config_mode}")

    if [[ -z "$campaign_id" ]] || [[ "$campaign_id" == "0" ]]; then
        error "Failed to plan campaign — no campaign ID returned"
        track "$run_key" "$config_mode" "$layer" "$model" "0" "failed" "No campaign ID returned from plan"
        return 1
    fi

    success "Planned — Campaign ID: ${campaign_id}"
    track "$run_key" "$config_mode" "$layer" "$model" "$campaign_id" "planned" ""

    # Phase 2: Run
    log "Running campaign ${campaign_id}..."
    track "$run_key" "$config_mode" "$layer" "$model" "$campaign_id" "running" ""

    if run_campaign "$campaign_id"; then
        track "$run_key" "$config_mode" "$layer" "$model" "$campaign_id" "completed" ""
    else
        warn "Campaign ${campaign_id} finished with issues"
        track "$run_key" "$config_mode" "$layer" "$model" "$campaign_id" "completed_warnings" ""
    fi

    # Phase 3: Export
    export_campaign "$campaign_id"

    # Store campaign_id for comparison
    echo "${run_key}=${campaign_id}" >> "${LOG_DIR}/campaign_ids.txt"

    success "Campaign ${campaign_id} finished"
    echo ""
}

# ── Main Execution ─────────────────────────────────────────────────────────

run_config_mode() {
    local config_mode="$1"  # "baseline" or "hardened"

    header "Config: ${config_mode^^}"
    log "Starting config ${config_mode} run..."

    # Start lab with appropriate hardening
    start_lab "$config_mode" || {
        error "Failed to start lab for config ${config_mode}"
        return 1
    }

    # Ensure all models are available
    ensure_models

    # Get filtered campaign list
    local campaigns=()
    mapfile -t campaigns < <(get_filtered_campaigns)

    if [[ ${#campaigns[@]} -eq 0 ]]; then
        warn "No campaigns match filters for config ${config_mode}"
        stop_lab
        return 0
    fi

    log "Will execute ${#campaigns[@]} campaign(s) for config ${config_mode}"
    echo ""

    local total=${#campaigns[@]}
    local current=0
    local errors=0

    for key in "${campaigns[@]}"; do
        current=$((current + 1))
        log "Progress: [${current}/${total}]"

        if ! execute_campaign "$key" "$config_mode"; then
            errors=$((errors + 1))
            warn "Campaign ${key} failed (${errors} error(s) so far)"
            # Continue with next campaign — don't abort the whole batch
        fi

        # Small cooldown between campaigns
        sleep 2
    done

    log "Config ${config_mode}: ${current} campaigns processed, ${errors} errors"

    stop_lab

    if [[ $errors -gt 0 ]]; then
        return 1
    fi
    return 0
}

generate_comparison_report() {
    header "Generating Comparison Report"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "[dry-run] Would generate comparison report"
        return 0
    fi

    {
        echo "=============================================="
        echo "  Norn Lab Experiment — Comparison Report"
        echo "  Generated: $(date)"
        echo "=============================================="
        echo ""

        # Per-campaign summary from tracker
        echo "── Campaign Execution Summary ──"
        echo ""
        printf "%-30s %-10s %-8s %-6s %s\n" "RUN_KEY" "CONFIG" "LAYER" "STATE" "CAMPAIGN_ID"
        printf "%-30s %-10s %-8s %-6s %s\n" "───────" "──────" "─────" "─────" "───────────"
        if [[ -f "${TRACK_FILE}" ]]; then
            tail -n +2 "${TRACK_FILE}" | while IFS=',' read -r key config layer model cid state started finished err; do
                printf "%-30s %-10s %-8s %-6s %s\n" "$key" "$config" "$layer" "$state" "$cid"
            done
        fi
        echo ""

        # Per-campaign metric summary from metric_aggregate table
        if [[ -f "${DB_FILE}" ]]; then
            echo "── Metric Summary ──"
            echo ""

            # Use tracker CSV to map campaign_ids to config_mode
            declare -A CID_MODE
            if [[ -f "${TRACK_FILE}" ]]; then
                while IFS=',' read -r run_key config_mode layer model cid state started finished err; do
                    [[ -n "$cid" ]] && [[ "$cid" != "campaign_id" ]] && CID_MODE["$cid"]="$config_mode"
                done < "${TRACK_FILE}"
            fi

            sqlite3 -column -header "${DB_FILE}" "
                SELECT
                    a.campaign_id AS cid,
                    c.name,
                    c.layer,
                    a.metric_id AS metric,
                    ROUND(a.mean, 4) AS mean,
                    ROUND(a.std_dev, 4) AS std_dev,
                    ROUND(a.ci95_lower, 4) AS ci95_lower,
                    ROUND(a.ci95_upper, 4) AS ci95_upper
                FROM metric_aggregate a
                JOIN campaign c ON c.id = a.campaign_id
                ORDER BY c.id, a.metric_id;
            " 2>/dev/null || echo "(no metrics data — campaigns may not have completed yet)"

            echo ""

            # Side-by-side baseline vs hardened comparison
            # Only generate when both modes were run; skip if single-mode
            if [[ "$MODE" == "all" ]]; then
                echo "── Baseline vs Hardened (by Model, Layer, Metric) ──"
                echo ""

            # Build simple comparison using tracker csv + metric_aggregate queries
            local model_tags=("qwen3.5:2b" "nemotron-3-nano:4b" "qwen3.5:4b" "gemma4:26b")
            local model_keys=("qwen35_2b" "nemotron4b" "qwen35_4b" "gemma4_26b")

            for layer in L1 L2 L3; do
                local -a metric_ids
                case "$layer" in
                    L1) metric_ids=("ASR" "FAR" "FRR") ;;
                    L2) metric_ids=("ASR-L2" "PSR@5" "TDS") ;;
                    L3) metric_ids=("UAR" "CTER" "KCCR") ;;
                esac

                echo "### ${layer} Metrics ###"
                echo ""

                for metric_id in "${metric_ids[@]}"; do
                    echo "--- ${metric_id} ---"
                    printf "%-15s %-12s %-12s %-10s\n" "MODEL" "BASELINE" "HARDENED" "DELTA"
                    printf "%-15s %-12s %-12s %-10s\n" "─────" "────────" "────────" "─────"

                    for i in "${!model_tags[@]}"; do
                        local model_tag="${model_tags[$i]}"
                        local model_key="${model_keys[$i]}"
                        local model_label="${model_tag%%:*}"
                        local base_val="N/A"
                        local hard_val="N/A"

                        if [[ -f "${TRACK_FILE}" ]]; then
                            local base_cid
                            base_cid=$(grep ",baseline,${layer},${model_key}," "${TRACK_FILE}" 2>/dev/null | tail -1 | cut -d, -f5 || echo "")
                            local hard_cid
                            hard_cid=$(grep ",hardened,${layer},${model_key}," "${TRACK_FILE}" 2>/dev/null | tail -1 | cut -d, -f5 || echo "")

                            [[ -n "$base_cid" ]] && base_val=$(sqlite3 "${DB_FILE}" \
                                "SELECT ROUND(mean, 4) FROM metric_aggregate WHERE campaign_id=${base_cid} AND metric_id='${metric_id}';" 2>/dev/null || echo "N/A")
                            [[ -n "$hard_cid" ]] && hard_val=$(sqlite3 "${DB_FILE}" \
                                "SELECT ROUND(mean, 4) FROM metric_aggregate WHERE campaign_id=${hard_cid} AND metric_id='${metric_id}';" 2>/dev/null || echo "N/A")
                        fi

                        if [[ "$base_val" != "N/A" ]] && [[ "$hard_val" != "N/A" ]]; then
                            local delta
                            delta=$(python -c "print(round(${hard_val} - ${base_val}, 4))" 2>/dev/null || echo "N/A")
                            printf "%-15s %-12s %-12s %-10s\n" "$model_label" "$base_val" "$hard_val" "$delta"
                        else
                            printf "%-15s %-12s %-12s %-10s\n" "$model_label" "$base_val" "$hard_val" "—"
                        fi
                    done
                    echo ""
                done
            done
            fi
        fi

        echo ""
        echo "── Output Files ──"
        echo "  Logs:        ${LOG_DIR}/"
        echo "  Tracker CSV: ${TRACK_FILE}"
        echo "  Exports:     ${NORN_DIR}/norn_exports/"
        echo "  Database:    ${DB_FILE}"

    } > "${SUMMARY_FILE}"

    success "Comparison report generated: ${SUMMARY_FILE}"
    cat "${SUMMARY_FILE}"
}

# ── Cleanup ────────────────────────────────────────────────────────────────

cleanup() {
    local exit_code=$?
    echo ""
    log "Cleaning up..."

    # Stop lab if it's running
    cd "${LAB_DIR}" 2>/dev/null && docker compose down --remove-orphans 2>/dev/null || true

    if [[ $exit_code -ne 0 ]] && [[ "$DRY_RUN" != "true" ]]; then
        warn "Experiment exited with code ${exit_code}"
        warn "Tracker: ${TRACK_FILE}"
        warn "Use --resume to continue from last successful campaign"
    fi

    log "Done."
    exit $exit_code
}

trap cleanup EXIT

# ── Entry Point ────────────────────────────────────────────────────────────

main() {
    banner

    log "Mode: ${MODE}"
    log "Filters — layers=${LAYERS:-all}, models=${MODELS:-all}"
    log "Database: ${DB_FILE}"
    log "Lab directory: ${LAB_DIR}"
    log "Log directory: ${LOG_DIR}"
    log "Quick mode: ${QUICK_MODE}"
    log "Resume: ${RESUME}"
    echo ""

    # ── Report-Only Fast Path ──
    if [[ "$REPORT_ONLY" == "true" ]]; then
        if [[ ! -f "${DB_FILE}" ]]; then
            error "Database not found: ${DB_FILE}"
            exit 1
        fi
        if ! command -v sqlite3 &>/dev/null; then
            error "sqlite3 required for report generation"
            exit 1
        fi
        MODE="all"  # force comparison section
        generate_comparison_report
        exit 0
    fi

    # Check prerequisites
    check_prerequisites || {
        error "Prerequisites not met — fix issues and retry"
        exit 1
    }

    if [[ "$CHECK_ONLY" == "true" ]]; then
        success "All prerequisites met — ready to run experiments"
        validate_all_configs
        exit 0
    fi

    # Initialize tracker
    init_tracker

    # Initialize database
    init_database || exit 1

    # Validate all campaign configs
    validate_all_configs || {
        error "Config validation failed"
        exit 1
    }

    # Show what we're about to run
    local campaigns=()
    mapfile -t campaigns < <(get_filtered_campaigns)
    local total_runs=${#campaigns[@]}

    if [[ "$MODE" == "all" ]]; then
        total_runs=$((total_runs * 2))
    fi

    header "Experiment Plan"
    log "Campaigns to execute: ${total_runs}"
    echo ""
    for key in "${campaigns[@]}"; do
        local yaml="${CAMPAIGN_FILES[$key]}"
        if [[ "$MODE" == "all" ]] || [[ "$MODE" == "baseline" ]]; then
            echo "  [baseline]  ${key} → ${yaml}"
        fi
        if [[ "$MODE" == "all" ]] || [[ "$MODE" == "hardened" ]]; then
            echo "  [hardened]  ${key} → ${yaml}"
        fi
    done
    echo ""

    if [[ "$DRY_RUN" == "true" ]]; then
        log "Dry-run complete — no campaigns executed"
        exit 0
    fi

    # Confirm execution
    echo -e "${YELLOW}About to execute ${total_runs} campaign runs.${NC}"
    echo -e "This will take approximately 2–4 hours depending on model inference speed."
    read -rp "Proceed? [y/N]: " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        log "Aborted by user"
        exit 0
    fi

    local start_ts
    start_ts=$(date +%s)

    # ── Config A: Baseline ──
    if [[ "$MODE" == "all" ]] || [[ "$MODE" == "baseline" ]]; then
        run_config_mode "baseline" || warn "Baseline run had errors — check logs"
    fi

    # ── Config B: Hardened ──
    if [[ "$MODE" == "all" ]] || [[ "$MODE" == "hardened" ]]; then
        run_config_mode "hardened" || warn "Hardened run had errors — check logs"
    fi

    local end_ts elapsed
    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))

    # ── Final Report ──
    header "Experiment Complete"
    success "Total elapsed time: ${hours}h ${minutes}m"
    echo ""

    generate_comparison_report

    echo ""
    success "Experiment finished. Results available in:"
    echo "  Report:  ${SUMMARY_FILE}"
    echo "  Tracker: ${TRACK_FILE}"
    echo "  Exports: ${NORN_DIR}/norn_exports/"
    echo "  Logs:    ${LOG_DIR}/"
}

main "$@"
