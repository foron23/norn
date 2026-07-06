#!/usr/bin/env bash
# ==============================================================================
# run_cloud_experiments.sh — Norn Cloud Lab Experiment Orchestrator
# ==============================================================================
# Executes the cloud-based A/B hardening comparison for the TFM thesis:
#   4 cloud models × 3 layers × 2 configs (baseline + hardened) = 24 runs
#   Models: nemotron-3-super:cloud, minimax-m2.7:cloud, qwen3.5:cloud, gemma4:31b-cloud
#
# All generation runs on ollama.com cloud. Only nomic-embed-text runs locally.
# The lab web app handles RAG pipeline, hardening gates, and L3 agent orchestration.
#
# Usage:
#   ./run_cloud_experiments.sh                  # Full experiment (A + B)
#   ./run_cloud_experiments.sh --mode baseline  # Only Config A
#   ./run_cloud_experiments.sh --mode hardened  # Only Config B
#   ./run_cloud_experiments.sh --check          # Validate prerequisites only
#   ./run_cloud_experiments.sh --quick          # Reduced replicas for smoke testing
#   ./run_cloud_experiments.sh --layers L1,L3   # Only specific layers
#   ./run_cloud_experiments.sh --models nemotron,qwen  # Only specific models
# ==============================================================================
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NORN_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LAB_DIR="${LAB_DIR:-${HOME}/Documents/master/TFM/lab}"
DB_FILE="${NORN_DIR}/norn_cloud.db"                           # separate DB from local experiments
CAMPAIGNS_DIR="${SCRIPT_DIR}"
LOG_DIR="${NORN_DIR}/experiment_logs/cloud"
TRACK_FILE="${LOG_DIR}/campaign_tracker.csv"
SUMMARY_FILE="${LOG_DIR}/summary_$(date +%Y%m%d_%H%M%S).txt"

# Shell colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

# Only embedding model needs to be local — generation runs on ollama.com cloud
REQUIRED_MODELS=("nomic-embed-text")

# Campaign YAML files — all 12 combinations (4 models × 3 layers)
declare -A CAMPAIGN_FILES=(
    ["l1_nemotron3_super"]="l1_nemotron3_super.yaml"
    ["l1_minimax_m27"]="l1_minimax_m27.yaml"
    ["l1_qwen35"]="l1_qwen35.yaml"
    ["l1_gemma4_31b"]="l1_gemma4_31b.yaml"
    ["l2_nemotron3_super"]="l2_nemotron3_super.yaml"
    ["l2_minimax_m27"]="l2_minimax_m27.yaml"
    ["l2_qwen35"]="l2_qwen35.yaml"
    ["l2_gemma4_31b"]="l2_gemma4_31b.yaml"
    ["l3_nemotron3_super"]="l3_nemotron3_super.yaml"
    ["l3_minimax_m27"]="l3_minimax_m27.yaml"
    ["l3_qwen35"]="l3_qwen35.yaml"
    ["l3_gemma4_31b"]="l3_gemma4_31b.yaml"
)

# Model tag mapping for report generation
declare -A MODEL_TAGS=(
    ["nemotron3_super"]="nemotron-3-super:cloud"
    ["minimax_m27"]="minimax-m2.5:cloud"
    ["qwen35"]="qwen3-coder-next:cloud"
    ["gemma4_31b"]="gemma4:31b-cloud"
)

# ── CLI Parsing ────────────────────────────────────────────────────────────

MODE="all"
QUICK_MODE=false
CHECK_ONLY=false
RESUME=false
LAYERS=""
MODELS=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --quick) QUICK_MODE=true; shift ;;
        --check) CHECK_ONLY=true; shift ;;
        --resume) RESUME=true; shift ;;
        --layers) LAYERS="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) error "Unknown option: $1"; usage; exit 1 ;;
    esac
done

usage() {
    cat << EOF
${BOLD}Norn Cloud Lab Experiment Orchestrator${NC}

Usage: $0 [OPTIONS]

${BOLD}Options:${NC}
  --mode baseline|hardened|all   Which config to run (default: all)
  --quick                        Smoke test with R=2 replicas
  --check                        Validate prerequisites only
  --resume                       Skip completed campaigns
  --layers L1,L2,L3              Filter specific layers
  --models nemotron,minimax,qwen,gemma4  Filter specific models
  --dry-run                      Show what would run without executing
  --help                         This message
EOF
}

# ── Logging ────────────────────────────────────────────────────────────────

mkdir -p "${LOG_DIR}"

header()  { echo -e "\n${BOLD}${CYAN}── $1 ──${NC}"; }
success() { echo -e "  ${GREEN}✓${NC} $1"; }
warn()    { echo -e "  ${YELLOW}⚠${NC} $1"; }
error()   { echo -e "  ${RED}✗${NC} $1"; }
log()     { echo "  $1"; }
banner()  { echo -e "${BOLD}╔══════════════════════════════════════════════╗
║  Norn Cloud Lab — Experiment Orchestrator    ║
║  4 models × 3 layers × 2 configs = 24 runs   ║
║  Generation: ollama.com cloud                ║
╚══════════════════════════════════════════════╝${NC}"; }

# ── Tracker ────────────────────────────────────────────────────────────────

init_tracker() {
    if [[ ! -f "${TRACK_FILE}" ]] || [[ "${RESUME}" == "false" ]]; then
        echo "run_key,config_mode,layer,model_name,campaign_id,state,started_at,finished_at,error_msg" > "${TRACK_FILE}"
    fi
}

track() {
    local run_key="$1" config_mode="$2" layer="$3" model="$4"
    local campaign_id="$5" state="$6" error_msg="${7:-}"
    local ts=$(date -Iseconds)
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

# ── Filtering ──────────────────────────────────────────────────────────────

get_filtered_campaigns() {
    local filtered=()
    for key in $(echo "${!CAMPAIGN_FILES[@]}" | tr ' ' '\n' | sort); do
        local layer=$(echo "$key" | cut -d_ -f1 | tr '[:lower:]' '[:upper:]')
        local model=$(echo "$key" | cut -d_ -f2-)

        # Layer filter
        if [[ -n "$LAYERS" ]]; then
            local match=false
            IFS=',' read -ra LAYER_LIST <<< "$LAYERS"
            for l in "${LAYER_LIST[@]}"; do
                [[ "${layer}" == "$l" ]] && match=true && break
            done
            [[ "$match" == "false" ]] && continue
        fi

        # Model filter
        if [[ -n "$MODELS" ]]; then
            local match=false
            IFS=',' read -ra MODEL_LIST <<< "$MODELS"
            for m in "${MODEL_LIST[@]}"; do
                [[ "${model}" == *"${m}"* ]] && match=true && break
            done
            [[ "$match" == "false" ]] && continue
        fi

        filtered+=("$key")
    done
    printf '%s\n' "${filtered[@]}"
}

# ── Prerequisites ──────────────────────────────────────────────────────────

check_prerequisites() {
    header "Checking Prerequisites"

    local errors=0

    for cmd in docker curl sqlite3 python; do
        if command -v "$cmd" &>/dev/null; then
            success "$cmd found"
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
        success "Lab docker-compose.yml found"
    else
        error "Lab docker-compose.yml not found at ${LAB_DIR}"
        errors=$((errors + 1))
    fi

    if [[ -d "${CAMPAIGNS_DIR}" ]]; then
        local count=$(ls "${CAMPAIGNS_DIR}"/l*_*.yaml 2>/dev/null | wc -l)
        success "${count} cloud campaign YAMLs found"
    else
        error "Cloud campaign directory not found: ${CAMPAIGNS_DIR}"
        errors=$((errors + 1))
    fi

    # Check ollama.com cloud API key in .env
    local api_key
    api_key=$(grep -oP 'OLLAMA_CLOUD_API_KEY=\K.*' "${LAB_DIR}/.env" 2>/dev/null | head -1 || echo "")
    if [[ -n "${api_key}" ]]; then
        success "OLLAMA_CLOUD_API_KEY set in .env"
    else
        warn "OLLAMA_CLOUD_API_KEY not set — cloud models will fail"
        warn "Get your key at: https://ollama.com/settings/keys"
        errors=$((errors + 1))
    fi

    if [[ $errors -gt 0 ]]; then
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
}

# ── Lab Lifecycle ──────────────────────────────────────────────────────────

start_lab() {
    local config_mode="$1"

    header "Starting Lab (${config_mode} config)"

    # Set hardening flags
    if [[ "$config_mode" == "hardened" ]]; then
        export L1_HARDENING=true
        export L2_HARDENING=true
        export L3_HARDENING=true
    else
        export L1_HARDENING=false
        export L2_HARDENING=false
        export L3_HARDENING=false
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        log "[dry-run] Would start lab with L1=${L1_HARDENING} L2=${L2_HARDENING} L3=${L3_HARDENING}"
        return 0
    fi

    cd "${LAB_DIR}"

    # Stop any previous instance
    docker compose down --remove-orphans 2>/dev/null || true
    sleep 2

    # Start with hardening env vars
    L1_HARDENING="${L1_HARDENING}" \
    L2_HARDENING="${L2_HARDENING}" \
    L3_HARDENING="${L3_HARDENING}" \
        docker compose up -d --wait 2>&1 || {
        error "Failed to start lab"
        return 1
    }

    # Wait for health checks
    local max_wait=60
    local waited=0
    while [[ $waited -lt $max_wait ]]; do
        if curl -s http://localhost:8085/v1/models >/dev/null 2>&1; then
            success "Lab healthy after ${waited}s"
            sleep 5  # extra settling time
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done

    error "Lab did not become healthy within ${max_wait}s"
    return 1
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

# ── Campaign Execution ─────────────────────────────────────────────────────

plan_campaign() {
    local yaml_path="$1"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "  [dry-run] Would plan: ${yaml_path}"
        echo "0"
        return 0
    fi

    local effective_yaml="${yaml_path}"

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
        log "  Quick mode: replicas=2"
    fi

    python -m norn.cli.main plan-campaign -c "${effective_yaml}" --db "${DB_FILE}" >&2
    local rc=$?

    [[ "$effective_yaml" != "${yaml_path}" ]] && rm -f "${effective_yaml}"

    if [[ $rc -ne 0 ]]; then
        echo "0"
        return 1
    fi

    sqlite3 "${DB_FILE}" "SELECT MAX(id) FROM campaign;" 2>/dev/null || echo "0"
}

run_campaign() {
    local campaign_id="$1"
    if [[ "$DRY_RUN" == "true" ]]; then
        log "  [dry-run] Would run campaign ${campaign_id}"
        return 0
    fi
    log "  Running campaign ${campaign_id}..."
    python -m norn.cli.main run-campaign -id "${campaign_id}" --db "${DB_FILE}" 2>&1
    local rc=$?
    [[ $rc -eq 0 ]] && success "  Campaign ${campaign_id} completed" \
                      || warn "  Campaign ${campaign_id} finished with warnings (rc=${rc})"
    return $rc
}

export_campaign() {
    local campaign_id="$1"
    if [[ "$DRY_RUN" == "true" ]]; then
        log "  [dry-run] Would export campaign ${campaign_id}"
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
    local layer=$(echo "$key" | cut -d_ -f1 | tr '[:lower:]' '[:upper:]')
    local model=$(echo "$key" | cut -d_ -f2-)
    local run_key="${config_mode}_${key}"

    header "Campaign: ${config_mode^^} / ${layer} / ${model}"
    log "YAML: ${yaml_path}"

    if [[ "$RESUME" == "true" ]] && is_completed "$run_key"; then
        success "Already completed — skipping"
        return 0
    fi

    track "$run_key" "$config_mode" "$layer" "$model" "" "planning" ""

    # Plan
    log "Planning..."
    local campaign_id=$(plan_campaign "${yaml_path}")
    if [[ -z "$campaign_id" ]] || [[ "$campaign_id" == "0" ]]; then
        error "Failed to plan campaign"
        track "$run_key" "$config_mode" "$layer" "$model" "0" "failed" "plan failed"
        return 1
    fi
    success "Planned — Campaign ID: ${campaign_id}"
    track "$run_key" "$config_mode" "$layer" "$model" "$campaign_id" "planned" ""

    # Run
    log "Running..."
    track "$run_key" "$config_mode" "$layer" "$model" "$campaign_id" "running" ""
    if run_campaign "$campaign_id"; then
        track "$run_key" "$config_mode" "$layer" "$model" "$campaign_id" "completed" ""
    else
        warn "Campaign ${campaign_id} finished with issues"
        track "$run_key" "$config_mode" "$layer" "$model" "$campaign_id" "completed_warnings" ""
    fi

    # Export
    export_campaign "$campaign_id"
    echo "${run_key}=${campaign_id}" >> "${LOG_DIR}/campaign_ids.txt"
    success "Campaign ${campaign_id} done"
}

run_config_mode() {
    local config_mode="$1"

    header "Config: ${config_mode^^}"
    log "Starting ${config_mode} run with cloud models..."

    # Start lab with appropriate hardening toggles
    start_lab "$config_mode" || { error "Failed to start lab"; return 1; }

    local campaigns=()
    mapfile -t campaigns < <(get_filtered_campaigns)
    if [[ ${#campaigns[@]} -eq 0 ]]; then
        warn "No campaigns match filters"
        stop_lab
        return 0
    fi

    log "Will execute ${#campaigns[@]} campaign(s)"
    local total=${#campaigns[@]}
    local current=0 errors=0

    for key in "${campaigns[@]}"; do
        current=$((current + 1))
        log "Progress: [${current}/${total}]"
        execute_campaign "$key" "$config_mode" || { errors=$((errors + 1)); warn "${errors} error(s) so far"; }
        sleep 2
    done

    log "Config ${config_mode}: ${current} processed, ${errors} errors"
    stop_lab
    return $(( errors > 0 ? 1 : 0 ))
}

# ── Validation ─────────────────────────────────────────────────────────────

validate_all_configs() {
    header "Validating Cloud Campaign Configs"
    local errors=0
    local campaigns=()
    mapfile -t campaigns < <(get_filtered_campaigns)

    for key in "${campaigns[@]}"; do
        local yaml="${CAMPAIGN_FILES[$key]}"
        local path="${CAMPAIGNS_DIR}/${yaml}"
        log "Validating ${yaml}..."
        if [[ "$DRY_RUN" == "true" ]]; then log "  [dry-run] Would validate"; continue; fi
        python -m norn.cli.main validate-config "${path}" >/dev/null 2>&1 && success "  ${yaml}" \
            || { error "  ${yaml}"; errors=$((errors + 1)); }
    done
    return $(( errors > 0 ? 1 : 0 ))
}

# ── Report ─────────────────────────────────────────────────────────────────

generate_comparison_report() {
    header "Generating Comparison Report"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "[dry-run] Would generate report"
        return 0
    fi

    {
        echo "=============================================="
        echo "  Norn Cloud Lab — Comparison Report"
        echo "  Generated: $(date)"
        echo "  Models: ollama.com cloud"
        echo "=============================================="
        echo ""

        echo "── Execution Summary ──"
        echo ""
        printf "%-35s %-10s %-8s %-6s %s\n" "RUN_KEY" "CONFIG" "LAYER" "STATE" "CID"
        printf "%-35s %-10s %-8s %-6s %s\n" "───────" "──────" "─────" "─────" "───"
        if [[ -f "${TRACK_FILE}" ]]; then
            tail -n +2 "${TRACK_FILE}" | while IFS=',' read -r key config layer model cid state started finished err; do
                printf "%-35s %-10s %-8s %-6s %s\n" "$key" "$config" "$layer" "$state" "$cid"
            done
        fi
        echo ""

        if [[ -f "${DB_FILE}" ]]; then
            echo "── Metric Summary ──"
            echo ""
            sqlite3 -column -header "${DB_FILE}" "
                SELECT a.campaign_id AS cid, c.name, c.layer,
                       a.metric_id AS metric,
                       ROUND(a.mean, 4) AS mean,
                       ROUND(a.ci95_lower, 4) AS ci95_l,
                       ROUND(a.ci95_upper, 4) AS ci95_u
                FROM metric_aggregate a
                JOIN campaign c ON c.id = a.campaign_id
                ORDER BY c.id, a.metric_id;
            " 2>/dev/null || echo "(no metrics — campaigns may not have completed)"
            echo ""

            if [[ "$MODE" == "all" ]]; then
                echo "── Baseline vs Hardened Comparison ──"
                echo ""

                local model_keys=("nemotron3_super" "minimax_m27" "qwen35" "gemma4_31b")
                local model_labels=("nemotron-3-super" "minimax-m2.5" "qwen3-coder-next" "gemma4-31b")

                for layer in L1 L2 L3; do
                    case "$layer" in
                        L1) metric_ids=("ASR" "FAR" "FRR") ;;
                        L2) metric_ids=("ASR-L2" "PSR@5" "TDS") ;;
                        L3) metric_ids=("UAR" "CTER" "KCCR") ;;
                    esac

                    echo "### ${layer} ###"
                    for metric_id in "${metric_ids[@]}"; do
                        echo "--- ${metric_id} ---"
                        printf "%-18s %-12s %-12s %-8s\n" "MODEL" "BASELINE" "HARDENED" "DELTA"
                        for i in "${!model_keys[@]}"; do
                            local mk="${model_keys[$i]}"
                            local ml="${model_labels[$i]}"
                            local base_cid=$(grep ",baseline,${layer},${mk}," "${TRACK_FILE}" 2>/dev/null | tail -1 | cut -d, -f5 || echo "")
                            local hard_cid=$(grep ",hardened,${layer},${mk}," "${TRACK_FILE}" 2>/dev/null | tail -1 | cut -d, -f5 || echo "")
                            local base_val="N/A"; local hard_val="N/A"
                            [[ -n "$base_cid" ]] && base_val=$(sqlite3 "${DB_FILE}" "SELECT ROUND(mean,4) FROM metric_aggregate WHERE campaign_id=${base_cid} AND metric_id='${metric_id}';" 2>/dev/null || echo "N/A")
                            [[ -n "$hard_cid" ]] && hard_val=$(sqlite3 "${DB_FILE}" "SELECT ROUND(mean,4) FROM metric_aggregate WHERE campaign_id=${hard_cid} AND metric_id='${metric_id}';" 2>/dev/null || echo "N/A")
                            if [[ "$base_val" != "N/A" ]] && [[ "$hard_val" != "N/A" ]]; then
                                local delta=$(python -c "print(round(${hard_val} - ${base_val}, 4))" 2>/dev/null || echo "—")
                                printf "%-18s %-12s %-12s %-8s\n" "$ml" "$base_val" "$hard_val" "$delta"
                            else
                                printf "%-18s %-12s %-12s %-8s\n" "$ml" "$base_val" "$hard_val" "—"
                            fi
                        done
                        echo ""
                    done
                done
            fi
        fi

        echo "── Output Files ──"
        echo "  Logs:    ${LOG_DIR}/"
        echo "  Tracker: ${TRACK_FILE}"
        echo "  DB:      ${DB_FILE}"
        echo "  Exports: ${NORN_DIR}/norn_exports/"
    } > "${SUMMARY_FILE}"

    success "Report: ${SUMMARY_FILE}"
    cat "${SUMMARY_FILE}"
}

# ── Main ───────────────────────────────────────────────────────────────────

cleanup() {
    echo ""
    log "Cleaning up..."
    cd "${LAB_DIR}" 2>/dev/null && docker compose down --remove-orphans 2>/dev/null || true
    log "Done."
}

trap cleanup EXIT

main() {
    banner
    log "Mode: ${MODE} | Quick: ${QUICK_MODE} | Resume: ${RESUME}"
    log "Filter — layers=${LAYERS:-all} models=${MODELS:-all}"
    log "DB: ${DB_FILE} | Configs: ${CAMPAIGNS_DIR}"

    if check_prerequisites; then
        success "Prerequisites OK"
    else
        error "Fix issues and retry"
        exit 1
    fi

    [[ "$CHECK_ONLY" == "true" ]] && { validate_all_configs; exit 0; }

    init_tracker
    init_database || exit 1
    validate_all_configs || { error "Config validation failed"; exit 1; }

    local campaigns=()
    mapfile -t campaigns < <(get_filtered_campaigns)
    local total_runs=${#campaigns[@]}
    [[ "$MODE" == "all" ]] && total_runs=$((total_runs * 2))

    header "Experiment Plan"
    log "${total_runs} campaign runs"
    for key in "${campaigns[@]}"; do
        local yaml="${CAMPAIGN_FILES[$key]}"
        [[ "$MODE" == "all" || "$MODE" == "baseline" ]] && echo "  [baseline]  ${key} → ${yaml}"
        [[ "$MODE" == "all" || "$MODE" == "hardened" ]] && echo "  [hardened]  ${key} → ${yaml}"
    done

    [[ "$DRY_RUN" == "true" ]] && { log "Dry-run complete"; exit 0; }

    echo -e "\n${YELLOW}About to execute ${total_runs} cloud campaign runs.${NC}"
    read -rp "Proceed? [y/N]: " confirm
    [[ ! "$confirm" =~ ^[Yy]$ ]] && { log "Aborted"; exit 0; }

    local start_ts=$(date +%s)

    [[ "$MODE" == "all" || "$MODE" == "baseline" ]] && run_config_mode "baseline"
    [[ "$MODE" == "all" || "$MODE" == "hardened" ]] && run_config_mode "hardened"

    local elapsed=$(($(date +%s) - start_ts))
    header "Experiment Complete"
    success "Elapsed: $((elapsed / 3600))h $(((elapsed % 3600) / 60))m"
    generate_comparison_report
}

main
