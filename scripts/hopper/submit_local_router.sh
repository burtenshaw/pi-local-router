#!/usr/bin/env bash
set -euo pipefail

LOGIN_HOST="${LOGIN_HOST:-hpc-cluster-hopper-login-node-1}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_USER="${REMOTE_USER:-$(ssh "$LOGIN_HOST" 'printf "%s" "$USER"')}"
REMOTE_BASE="${REMOTE_BASE:-/fsx/${REMOTE_USER}/pi-local-router}"
REMOTE_ROOT="${REMOTE_ROOT:-${REMOTE_BASE}/current}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-${REMOTE_BASE}/logs}"

ssh "$LOGIN_HOST" "mkdir -p '$REMOTE_ROOT' '$REMOTE_LOG_ROOT'"
rsync -az --delete \
  --exclude ".git" \
  --exclude "node_modules" \
  --exclude ".venv" \
  --exclude "__pycache__" \
  --exclude ".pytest_cache" \
  "$LOCAL_ROOT/" "$LOGIN_HOST:$REMOTE_ROOT/"

SBATCH_EXPORTS="ALL,RUN_ROOT=$REMOTE_ROOT,LOG_ROOT=$REMOTE_LOG_ROOT"
for name in \
  HF_TOKEN \
  PYTHON_BIN \
  UV_BIN \
  LLAMA_SERVER_BIN \
  LLAMA_MODEL \
  LLAMA_HF_REPO \
  LLAMA_HF_FILE \
  LLAMA_CTX_SIZE \
  LLAMA_N_GPU_LAYERS \
  LLAMA_EXTRA_ARGS \
  LLAMA_SERVER_PORT \
  LOCAL_ROUTER_MODEL_ID \
  LOCAL_ROUTER_PORT \
  LOCAL_ROUTER_BACKEND_BASE_URL \
  LOCAL_ROUTER_MAX_TOKENS \
  LOCAL_ROUTER_ENTROPY_THRESHOLD \
  LOCAL_ROUTER_TOP1_THRESHOLD \
  LOCAL_ROUTER_CONFIDENCE_THRESHOLD \
  LOCAL_ROUTER_TOP_LOGPROBS \
  LOCAL_ROUTER_PROBE_MAX_CONTEXT_CHARS \
  LOCAL_ROUTER_PROBE_MAX_MESSAGE_CHARS \
  LOCAL_ROUTER_MAX_CONCURRENCY \
  LOCAL_ROUTER_DECISION_CACHE_SIZE \
  LOCAL_ROUTER_REBUILD_VENV
do
  value="${!name:-}"
  if [[ -n "$value" ]]; then
    SBATCH_EXPORTS+=",$name=$value"
  fi
done

SBATCH_ARGS=(
  --job-name="${SBATCH_JOB_NAME:-pi-local-router}"
  --partition="${SBATCH_PARTITION:-hopper-extra}"
  --qos="${SBATCH_QOS:-normal}"
  --nodes="${SBATCH_NODES:-1}"
  --ntasks=1
  --cpus-per-task="${SBATCH_CPUS_PER_TASK:-16}"
  --gres="${SBATCH_GRES:-gpu:h100:1}"
  --mem="${SBATCH_MEM:-240G}"
  --time="${SBATCH_TIME:-04:00:00}"
  --output="${REMOTE_LOG_ROOT}/%x-%j.out"
  --error="${REMOTE_LOG_ROOT}/%x-%j.err"
  --export="$SBATCH_EXPORTS"
  --parsable
)

ssh "$LOGIN_HOST" "cd '$REMOTE_ROOT' && /opt/slurm/bin/sbatch ${SBATCH_ARGS[*]@Q} scripts/hopper/run_local_router.sbatch"
