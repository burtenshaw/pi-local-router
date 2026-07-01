#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROUTER_PI_MODEL="${ROUTER_PI_MODEL:-local-router/local-to-hf}"
ROUTER_LOCAL_BASE_URL="${ROUTER_LOCAL_BASE_URL:-http://127.0.0.1:8080/v1}"

export ROUTER_LOCAL_BASE_URL

exec pi -e "$ROOT" --model "$ROUTER_PI_MODEL" "$@"
