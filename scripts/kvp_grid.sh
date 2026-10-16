#!/usr/bin/env bash
set -euo pipefail
cd /workspace/kv
for b in 128 256; do
  echo "== kvp b$b $(date +%T)"
  uv run --project ml-learning-to-evict python evict_eval.py --method kvp --budget $b 2>&1 | tee logs/grid_kvp_b$b.log
done
echo "== kvp grid done $(date +%T)"
