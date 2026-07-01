#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:?RUN_ROOT must be set by submit_local_router.sh}"
LOG_ROOT="${LOG_ROOT:-$RUN_ROOT/logs}"
VENV_DIR="${VENV_DIR:-$RUN_ROOT/.venv}"
HF_HOME="${HF_HOME:-$RUN_ROOT/.cache/huggingface}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUN_ROOT/.cache/xdg}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$RUN_ROOT/.cache/uv}"
LLAMA_SERVER_PORT="${LLAMA_SERVER_PORT:-8081}"
LOCAL_ROUTER_PORT="${LOCAL_ROUTER_PORT:-8080}"
LOCAL_ROUTER_BACKEND_BASE_URL="${LOCAL_ROUTER_BACKEND_BASE_URL:-http://127.0.0.1:${LLAMA_SERVER_PORT}/v1}"

mkdir -p "$LOG_ROOT" "$HF_HOME" "$XDG_CACHE_HOME" "$UV_CACHE_DIR"
cd "$RUN_ROOT"

export HF_HOME
export XDG_CACHE_HOME
export UV_CACHE_DIR

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.11)"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

if [[ -z "${UV_BIN:-}" ]]; then
  if [[ -x "$HOME/.local/bin/uv" ]]; then
    UV_BIN="$HOME/.local/bin/uv"
  elif command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
  fi
fi

if [[ "${LOCAL_ROUTER_REBUILD_VENV:-0}" == "1" ]]; then
  rm -rf "$VENV_DIR"
fi

if [[ -n "${UV_BIN:-}" ]]; then
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$UV_BIN" venv --python "$PYTHON_BIN" "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  "$UV_BIN" pip install -r requirements.txt
else
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
fi

if [[ -z "${LLAMA_SERVER_BIN:-}" ]]; then
  if command -v llama-server >/dev/null 2>&1; then
    LLAMA_SERVER_BIN="$(command -v llama-server)"
  else
    echo "llama-server not found. Set LLAMA_SERVER_BIN or add llama.cpp to PATH." >&2
    exit 1
  fi
fi

LLAMA_ARGS=(
  --host 0.0.0.0
  --port "$LLAMA_SERVER_PORT"
  --jinja
  --logits-all
  --ctx-size "${LLAMA_CTX_SIZE:-32768}"
  --n-gpu-layers "${LLAMA_N_GPU_LAYERS:-999}"
)

if [[ -n "${LLAMA_MODEL:-}" ]]; then
  LLAMA_ARGS+=(--model "$LLAMA_MODEL")
elif [[ -n "${LLAMA_HF_REPO:-}" && -n "${LLAMA_HF_FILE:-}" ]]; then
  LLAMA_ARGS+=(--hf-repo "$LLAMA_HF_REPO" --hf-file "$LLAMA_HF_FILE")
else
  echo "Set LLAMA_MODEL or both LLAMA_HF_REPO and LLAMA_HF_FILE." >&2
  exit 1
fi

if [[ -n "${LLAMA_EXTRA_ARGS:-}" ]]; then
  # Intentional word splitting so callers can pass extra llama-server flags.
  # shellcheck disable=SC2206
  EXTRA_ARGS=( $LLAMA_EXTRA_ARGS )
  LLAMA_ARGS+=("${EXTRA_ARGS[@]}")
fi

"$LLAMA_SERVER_BIN" "${LLAMA_ARGS[@]}" >"$LOG_ROOT/llama-server-${SLURM_JOB_ID:-manual}.log" 2>&1 &
LLAMA_PID=$!
trap 'kill "$LLAMA_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${LLAMA_SERVER_PORT}/v1/models" >/tmp/pi-local-router-models.json 2>/dev/null; then
    break
  fi
  if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
    echo "llama-server exited before becoming ready" >&2
    tail -n 80 "$LOG_ROOT/llama-server-${SLURM_JOB_ID:-manual}.log" >&2 || true
    exit 1
  fi
  sleep 5
done

if ! curl -fsS "http://127.0.0.1:${LLAMA_SERVER_PORT}/v1/models" >/tmp/pi-local-router-models.json; then
  echo "timed out waiting for llama-server /v1/models" >&2
  tail -n 80 "$LOG_ROOT/llama-server-${SLURM_JOB_ID:-manual}.log" >&2 || true
  exit 1
fi

if [[ -z "${LOCAL_ROUTER_MODEL_ID:-}" ]]; then
  LOCAL_ROUTER_MODEL_ID="$(python - <<'PY'
import json
with open("/tmp/pi-local-router-models.json") as handle:
    payload = json.load(handle)
print(payload["data"][0]["id"])
PY
)"
fi

python -m local_router.server \
  --host 0.0.0.0 \
  --port "$LOCAL_ROUTER_PORT" \
  --model-id "$LOCAL_ROUTER_MODEL_ID" \
  --backend-base-url "$LOCAL_ROUTER_BACKEND_BASE_URL" \
  --max-local-tokens "${LOCAL_ROUTER_MAX_TOKENS:-512}" \
  --entropy-threshold "${LOCAL_ROUTER_ENTROPY_THRESHOLD:-0.12}" \
  --top1-threshold "${LOCAL_ROUTER_TOP1_THRESHOLD:-0.95}" \
  --confidence-threshold "${LOCAL_ROUTER_CONFIDENCE_THRESHOLD:-0.97}" \
  --top-logprobs "${LOCAL_ROUTER_TOP_LOGPROBS:-20}" \
  --probe-max-context-chars "${LOCAL_ROUTER_PROBE_MAX_CONTEXT_CHARS:-12000}" \
  --probe-max-message-chars "${LOCAL_ROUTER_PROBE_MAX_MESSAGE_CHARS:-4000}" \
  --max-concurrency "${LOCAL_ROUTER_MAX_CONCURRENCY:-1}" \
  --decision-cache-size "${LOCAL_ROUTER_DECISION_CACHE_SIZE:-128}"
