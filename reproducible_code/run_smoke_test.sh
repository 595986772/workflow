#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

export MPLBACKEND=Agg
export MPLCONFIGDIR="$ROOT_DIR/.matplotlib"

cd "$ROOT_DIR"

"$PYTHON_BIN" "$ROOT_DIR/main.py" \
  -folder smoke_test \
  -alg nearestserver_prev_servers_plus_service_per_serverDQN \
  -nuser 3 \
  -nserver 3 \
  -nservice 3 \
  -ntasks 6 \
  -nepisode 8 \
  -batch_size 8 \
  -min_experiences 8 \
  -filling_steps 1 \
  -steps_to_updates 1 \
  -max_explore 8 \
  -seed 2 \
  -nruns 1
