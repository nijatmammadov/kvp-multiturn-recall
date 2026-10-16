#!/usr/bin/env bash
# Experiment batch v4 eval runs (one GPU, resumable). usage: scripts/exp_v4.sh defence|ft|ctl
#   defence: needs only the original agents. First-N protection baseline on the seed-0 set, and the "needle later"
#            set (2 filler turns before the needle) for full / kvp / h2o / streaming / streaming+first64.
#   ft:      needs the warm-started agents (oasst2_recall_ft_0_5b): kvp_ft on seed-0 (b256, b128), on lead2 (b256),
#            and kvp_ft with score pooling (b256).
#   ctl:     needs the control agents (oasst2_ctl_ft_0_5b): kvp_ftctl on seed-0, b256.
set -euo pipefail

PART=${1:?usage: exp_v4.sh defence|ft|ctl}
REPO=$(cd "$(dirname "$0")/.." && pwd)
P=/workspace/kv/ml-learning-to-evict
A=$P/agents/grouped/oasst2_0_5b
A_FT=$P/agents/grouped/oasst2_recall_ft_0_5b
A_CTL=$P/agents/grouped/oasst2_ctl_ft_0_5b
D0=$REPO/results/mt_niah.jsonl
DL=$REPO/results/mt_niah_lead2.jsonl
OUT0=$REPO/results/logs_eval_v4
OUTL=$REPO/results/logs_eval_v4_lead2
mkdir -p "$OUT0" "$OUTL"

check_agents() {  # dir logname: 48 agents, each exactly one run that reached step 1500, and no failed jobs
  local dir=$1 log=/workspace/kv/logs/$2 final n
  final=$(printf '%012d.pth' 1500)
  [ ! -f "$log/FAILED.txt" ] || { echo "$log/FAILED.txt exists: fine-tune incomplete" >&2; exit 1; }
  for L in $(seq 0 23); do for H in 0 1; do
    n=$(find "$dir/$(printf 'layer_%06d/kv_head_%03d' "$L" "$H")" -path "*/checkpoints/$final" 2>/dev/null | wc -l)
    [ "$n" -eq 1 ] || { echo "L$L H$H: $n finished runs in $dir (need exactly 1)" >&2; exit 1; }
  done; done
  [ "$(find "$dir" -name best_ckpt.pth | wc -l)" -eq 48 ] || { echo "need 48 best_ckpt.pth in $dir" >&2; exit 1; }
}

run() {
  local data=$1 out=$2 agents=$3; shift 3
  echo "== $(basename "$data") $* $(date -u +%H:%M:%S)"
  uv run --project "$P" python "$REPO/eval/evict_eval.py" --data "$data" --agents_dir "$agents" --out_dir "$out" "$@"
}

case $PART in
  defence)
    [ -f "$DL" ] || { echo "missing $DL (generate it first with eval/mt_niah.py --lead 2)" >&2; exit 1; }
    for b in 256 128; do run "$D0" "$OUT0" "$A" --method streaming --budget $b --protect_first 64 --tag first64; done
    run "$DL" "$OUTL" "$A" --method full --budget 0
    run "$DL" "$OUTL" "$A" --method kvp --budget 256
    run "$DL" "$OUTL" "$A" --method h2o --budget 256
    run "$DL" "$OUTL" "$A" --method streaming --budget 256
    run "$DL" "$OUTL" "$A" --method streaming --budget 256 --protect_first 64 --tag first64
    ;;
  ft)
    check_agents "$A_FT" train_0_5b_ft
    for b in 256 128; do run "$D0" "$OUT0" "$A_FT" --method kvp --budget $b --tag ft; done
    run "$DL" "$OUTL" "$A_FT" --method kvp --budget 256 --tag ft
    run "$D0" "$OUT0" "$A_FT" --method kvp --budget 256 --pool_window 4 --tag ft_pool4
    ;;
  ctl)
    check_agents "$A_CTL" train_0_5b_ftctl
    run "$D0" "$OUT0" "$A_CTL" --method kvp --budget 256 --tag ftctl
    ;;
  *) echo "unknown part $PART" >&2; exit 1 ;;
esac
echo "== $PART done $(date -u +%H:%M:%S)"
