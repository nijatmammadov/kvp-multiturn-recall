#!/usr/bin/env bash
# Experiment batch v6 eval runs (one GPU, resumable). usage: scripts/exp_v6.sh check|breadth|breadth_turn
#   check:   GPU regression + smoke (new code must reproduce v5 kvp_ft answers; smoke for --protect_heads).
#   breadth: words q1, kvp_ft b=256, needle force-kept in n = 6, 12, 24, 48 random heads, k = 4, 8.
#   breadth_turn (amendment 1): same, but the whole needle user turn is force-kept; n = 48 runs first and the
#            manipulation check (>= 0.80, conditioned on full-cache correct) must pass before n = 6, 12, 24 run.
set -euo pipefail

PART=${1:?usage: exp_v6.sh check|breadth|breadth_turn}
REPO=$(cd "$(dirname "$0")/.." && pwd)
P=/workspace/kv/ml-learning-to-evict
A_FT=$REPO/agents/oasst2_recall_ft_0_5b
DW=$REPO/results/mt_niah_words_lead2_q1.jsonl
OUTB=$REPO/results/logs_eval_v6_breadth

n=$(find "$A_FT" -name best_ckpt.pth | wc -l)
[ "$n" -eq 48 ] || { echo "need 48 best_ckpt.pth in $A_FT, found $n" >&2; exit 1; }

run() {
  local data=$1 out=$2; shift 2
  echo "== $(basename "$data") $* $(date -u +%H:%M:%S)"
  uv run --project "$P" python "$REPO/eval/evict_eval.py" --data "$data" --agents_dir "$A_FT" --out_dir "$out" "$@"
}

case $PART in
  check)
    C=$REPO/logs_check_v6
    [ ! -e "$C" ] || { echo "$C exists; move it away first" >&2; exit 1; }
    run "$DW" "$C" --method kvp --budget 256 --tag ft --ks 4 --limit 10
    python3 - "$C/kvp_ft_b256_bf16.jsonl" "$REPO/results/logs_eval_v5_words_q1/kvp_ft_b256_bf16.jsonl" <<'EOF'
import json, sys
new = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[1])}
old = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[2])}
same = sum(new[i]["answer"] == old[i]["answer"] for i in new)
print(f"regression: {same}/{len(new)} answers identical to v5 kvp_ft")
if same < len(new) - 1:
    sys.exit("REGRESSION FAILED: the v6 code changed the kvp path")
EOF
    run "$DW" "$C" --method kvp --budget 256 --protect_needle value --protect_heads 48 --tag ft_heads48 --ks 4 --limit 3
    run "$DW" "$C" --method kvp --budget 256 --protect_needle value --protect_heads 6 --tag ft_heads6 --ks 4 --limit 3
    python3 - "$C" <<'EOF'
import json, sys
d = sys.argv[1]
r48 = [json.loads(l) for l in open(f"{d}/kvp_ft_heads48_b256_bf16.jsonl")]
r6 = [json.loads(l) for l in open(f"{d}/kvp_ft_heads6_b256_bf16.jsonl")]
assert all(r["needle_heads_full_at_answer"] == 1.0 for r in r48), [r["needle_heads_full_at_answer"] for r in r48]
assert all(r["needle_heads_full_at_answer"] >= 6 / 48 - 1e-9 for r in r6), [r["needle_heads_full_at_answer"] for r in r6]
print("smoke OK: heads48 holds the needle in all heads, heads6 in >= 6")
EOF
    ;;
  breadth)
    for k in 6 12 24 48; do
      run "$DW" "$OUTB" --method kvp --budget 256 --protect_needle value --protect_heads $k --tag ft_heads$k --ks 4,8
    done
    ;;
  breadth_turn)
    run "$DW" "$OUTB" --method kvp --budget 256 --protect_needle turn --protect_heads 48 --tag ft_turn_heads48 --ks 4,8
    python3 - "$OUTB/kvp_ft_turn_heads48_b256_bf16.jsonl" "$REPO/results/logs_eval_v5_words_q1/full_b0_bf16.jsonl" <<'PY'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1])]
ok = {json.loads(l)["id"] for l in open(sys.argv[2]) if json.loads(l)["correct"]}
rs = [r for r in recs if r["id"] in ok]
acc = sum(r["correct"] for r in rs) / len(rs)
print(f"B2' (turn, n=48): {sum(r['correct'] for r in rs)}/{len(rs)} = {acc:.3f}")
if acc < 0.80:
    sys.exit("B2' FAILED: whole needle turn kept in all heads is still below 0.80; stop and report")
PY
    for k in 6 12 24; do
      run "$DW" "$OUTB" --method kvp --budget 256 --protect_needle turn --protect_heads $k --tag ft_turn_heads$k --ks 4,8
    done
    ;;
  *) echo "unknown part $PART" >&2; exit 1 ;;
esac
echo "== $PART done $(date -u +%H:%M:%S)"
