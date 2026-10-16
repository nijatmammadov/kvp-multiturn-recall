#!/usr/bin/env bash
# Eval-only experiment batch v2 (sequential, one GPU). Resumable: each run skips dialogues already in its JSONL.
#   1. KVP with agent positions rescaled into the trained range (T=600; sensitivity T=400 at b=256)
#   2. Standard H2O: half of the budget is a recent window (original H2O split)
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
P=/workspace/kv/ml-learning-to-evict
A=$P/agents/grouped/oasst2_0_5b
OUT=$REPO/results/logs_eval_v2
mkdir -p "$OUT"

run() {
  echo "== $* $(date -u +%H:%M:%S)"
  uv run --project "$P" python "$REPO/eval/evict_eval.py" --data "$REPO/results/mt_niah.jsonl" \
    --agents_dir "$A" --out_dir "$OUT" "$@"
}

for b in 256 128; do run --method kvp --budget $b --pos_scale_to 600 --tag pos600; done
run --method kvp --budget 256 --pos_scale_to 400 --tag pos400
for b in 256 128; do run --method h2o_std --budget $b; done
echo "== all done $(date -u +%H:%M:%S)"
