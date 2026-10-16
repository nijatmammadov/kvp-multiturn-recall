#!/usr/bin/env bash
# Experiment batch v5 eval runs (one GPU, resumable). usage: scripts/exp_v5.sh gen|words|digits
#   gen:     build the word-needle set (invented pet name + city, 2 filler turns before the needle).
#   words:   word set: full, kvp, h2o, streaming, kvp_ft (+ pool4/pool8) at b=256, kvp_ft (+ pool4/pool8) at b=128.
#   digits:  open v4 items on the digit sets: kvp_ft on lead-2, kvp_ft + pool4/pool8 on seed-0 and lead-2 (b=256).
set -euo pipefail

PART=${1:?usage: exp_v5.sh gen|words|digits}
REPO=$(cd "$(dirname "$0")/.." && pwd)
P=/workspace/kv/ml-learning-to-evict
A=$P/agents/grouped/oasst2_0_5b
A_FT=$REPO/agents/oasst2_recall_ft_0_5b
DW=$REPO/results/mt_niah_words_lead2_q1.jsonl
D0=$REPO/results/mt_niah.jsonl
DL=$REPO/results/mt_niah_lead2.jsonl
OUTW=$REPO/results/logs_eval_v5_words_q1
OUT0=$REPO/results/logs_eval_v4
OUTL=$REPO/results/logs_eval_v4_lead2
mkdir -p "$OUTW" "$OUT0" "$OUTL"

check_agents() {
  local n
  n=$(find "$1" -name best_ckpt.pth | wc -l)
  [ "$n" -eq 48 ] || { echo "need 48 best_ckpt.pth in $1, found $n" >&2; exit 1; }
}

run() {
  local data=$1 out=$2 agents=$3; shift 3
  echo "== $(basename "$data") $* $(date -u +%H:%M:%S)"
  uv run --project "$P" python "$REPO/eval/evict_eval.py" --data "$data" --agents_dir "$agents" --out_dir "$out" "$@"
}

case $PART in
  gen)
    [ ! -f "$DW" ] || { echo "$DW exists; not overwriting" >&2; exit 1; }
    cd "$REPO" && uv run --project "$P" python eval/mt_niah.py --needle words --seed 4 --lead 2 --out "$DW"
    ;;
  words)
    [ -f "$DW" ] || { echo "missing $DW: run 'exp_v5.sh gen' first" >&2; exit 1; }
    check_agents "$A"; check_agents "$A_FT"
    run "$DW" "$OUTW" "$A" --method full --budget 0
    python3 - "$OUTW/full_b0_bf16.jsonl" <<'EOF'
import json, sys
from collections import defaultdict
c = defaultdict(list)
for l in open(sys.argv[1]):
    r = json.loads(l); c[r["k"]].append(r["correct"])
print("full cache (amendment 2: analysis conditions on these being correct):",
      {k: f"{sum(v)}/{len(v)}" for k, v in sorted(c.items())})
EOF
    run "$DW" "$OUTW" "$A" --method kvp --budget 256
    run "$DW" "$OUTW" "$A" --method h2o --budget 256
    run "$DW" "$OUTW" "$A" --method streaming --budget 256
    for b in 256 128; do
      run "$DW" "$OUTW" "$A_FT" --method kvp --budget $b --tag ft
      run "$DW" "$OUTW" "$A_FT" --method kvp --budget $b --pool_window 4 --tag ft_pool4
      run "$DW" "$OUTW" "$A_FT" --method kvp --budget $b --pool_window 8 --tag ft_pool8
    done
    ;;
  digits)
    check_agents "$A_FT"
    run "$DL" "$OUTL" "$A_FT" --method kvp --budget 256 --tag ft
    for w in 4 8; do
      run "$D0" "$OUT0" "$A_FT" --method kvp --budget 256 --pool_window $w --tag ft_pool$w
      run "$DL" "$OUTL" "$A_FT" --method kvp --budget 256 --pool_window $w --tag ft_pool$w
    done
    ;;
  *) echo "unknown part $PART" >&2; exit 1 ;;
esac
echo "== $PART done $(date -u +%H:%M:%S)"
