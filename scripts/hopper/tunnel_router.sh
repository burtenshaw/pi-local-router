#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <slurm-job-id> [local-port] [remote-port]" >&2
  exit 2
fi

JOB_ID="$1"
LOCAL_PORT="${2:-8080}"
REMOTE_PORT="${3:-8080}"
LOGIN_HOST="${LOGIN_HOST:-hpc-cluster-hopper-login-node-1}"
TUNNEL_WAIT_SECONDS="${TUNNEL_WAIT_SECONDS:-900}"
ROUTER_READY_WAIT_SECONDS="${ROUTER_READY_WAIT_SECONDS:-1800}"

deadline=$((SECONDS + TUNNEL_WAIT_SECONDS))
NODE=""
while [[ $SECONDS -le $deadline ]]; do
  row="$(ssh "$LOGIN_HOST" "/opt/slurm/bin/squeue -j '$JOB_ID' -h -o '%T|%N|%R'" | tail -1)"
  if [[ -z "$row" ]]; then
    echo "job $JOB_ID is not in squeue; it may have exited before the server started" >&2
    ssh "$LOGIN_HOST" "/opt/slurm/bin/sacct -j '$JOB_ID' --format=JobID,JobName%24,State,ExitCode,Elapsed,NodeList%32 -n | tail -8" >&2 || true
    exit 1
  fi

  IFS='|' read -r state node reason <<<"$row"
  if [[ "$state" == "RUNNING" && -n "$node" && "$node" != "(null)" ]]; then
    NODE="$node"
    break
  fi

  echo "job $JOB_ID state=$state node=${node:-none} reason=${reason:-none}; waiting..." >&2
  sleep 10
done

if [[ -z "$NODE" ]]; then
  echo "timed out waiting for job $JOB_ID to receive a node" >&2
  exit 1
fi

deadline=$((SECONDS + ROUTER_READY_WAIT_SECONDS))
HEALTH_URL="http://${NODE}:${REMOTE_PORT}/health"
echo "job $JOB_ID is running on $NODE; waiting for router at ${HEALTH_URL}" >&2
while [[ $SECONDS -le $deadline ]]; do
  if ssh "$LOGIN_HOST" "curl -fsS --max-time 3 '$HEALTH_URL' >/dev/null"; then
    break
  fi
  echo "router is not listening yet on ${NODE}:${REMOTE_PORT}; waiting..." >&2
  sleep 10
done

if [[ $SECONDS -gt $deadline ]]; then
  echo "timed out waiting for router health endpoint at $HEALTH_URL" >&2
  exit 1
fi

echo "forwarding localhost:${LOCAL_PORT} -> ${NODE}:${REMOTE_PORT} through ${LOGIN_HOST}" >&2
ssh -N -L "${LOCAL_PORT}:${NODE}:${REMOTE_PORT}" "$LOGIN_HOST"
