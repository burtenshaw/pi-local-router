#!/usr/bin/env bash
set -euo pipefail

LOGIN_HOST="${LOGIN_HOST:-hpc-cluster-hopper-login-node-1}"

ssh -o BatchMode=yes -o ConnectTimeout=8 "$LOGIN_HOST" '
  set -euo pipefail
  hostname
  /opt/slurm/bin/sinfo -h -p "${SBATCH_PARTITION:-hopper-extra}" -o "%P %a nodes=%D gres=%G"
  /opt/slurm/bin/squeue -h -p "${SBATCH_PARTITION:-hopper-extra}" -u "$USER" -o "%i %T %M %N %j" || true
'
