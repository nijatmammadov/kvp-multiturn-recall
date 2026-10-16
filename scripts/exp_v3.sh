#!/usr/bin/env bash
# Eval-only experiment batch v3 (sequential, one GPU, resumable).
#   Part A (seed-0 dialogues): fragmentation fix (score pooling), needle-protection upper bounds, TOVA.
#   Part B (seed-1 dialogues, 140 per k): more data for the main comparison.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
P=/workspace/kv/ml-learning-to-evict
A=$P/agents/grouped/oasst2_0_5b
OUT_A=$REPO/results/logs_eval_v3
OUT_B=$REPO/results/logs_eval_v3_seed1
DATA_B=$REPO/results/mt_niah_seed1.jsonl
mkdir -p "$OUT_A" "$OUT_B"

run() {
  local data=$1 out=$2; shift 2
  echo "== $* $(date -u +%H:%M:%S)"
  uv run --project "$P" python "$REPO/eval/evict_eval.py" --data "$data" --agents_dir "$A" --out_dir "$out" "$@"
}

# ---- Part A: seed-0 dialogues (paired with v1/v2) ----
D=$REPO/results/mt_niah.jsonl
run "$D" "$OUT_A" --method kvp --budget 256 --pool_window 4 --tag pool4
run "$D" "$OUT_A" --method kvp --budget 128 --pool_window 4 --tag pool4
run "$D" "$OUT_A" --method kvp --budget 256 --protect_needle value --tag protectvalue
run "$D" "$OUT_A" --method kvp --budget 256 --protect_needle turn --tag protectturn
run "$D" "$OUT_A" --method kvp --budget 256 --pool_window 2 --tag pool2
run "$D" "$OUT_A" --method h2o --budget 256 --pool_window 4 --tag pool4
for b in 256 128; do run "$D" "$OUT_A" --method tova --budget $b; done

# ---- Part B: seed-1 dialogues ----
[ -f "$DATA_B" ] || { echo "missing $DATA_B (generate it first with eval/mt_niah.py)" >&2; exit 1; }
run "$DATA_B" "$OUT_B" --method full --budget 0 --limit 40
run "$DATA_B" "$OUT_B" --method kvp --budget 256
run "$DATA_B" "$OUT_B" --method kvp --budget 256 --pool_window 4 --tag pool4
run "$DATA_B" "$OUT_B" --method h2o --budget 256
run "$DATA_B" "$OUT_B" --method streaming --budget 256
echo "== all done $(date -u +%H:%M:%S)"
