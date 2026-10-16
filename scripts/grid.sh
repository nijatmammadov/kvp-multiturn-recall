#!/usr/bin/env bash
set -euo pipefail
cd /workspace/kv
mkdir -p logs
echo "== smoke $(date +%T)"
uv run --project ml-learning-to-evict python evict_eval.py --method h2o --budget 128 --limit 2 --out_dir /workspace/kv/logs_smoke 2>&1 | tee logs/grid_smoke_h2o_b128.log
for cfg in "full 0" "streaming 128" "streaming 256" "h2o 128" "h2o 256"; do
  set -- $cfg
  echo "== $1 b$2 $(date +%T)"
  uv run --project ml-learning-to-evict python evict_eval.py --method $1 --budget $2 2>&1 | tee logs/grid_$1_b$2.log
done
echo "== grid done $(date +%T)"
