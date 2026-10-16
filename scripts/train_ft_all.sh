#!/usr/bin/env bash
# Warm-start fine-tune of all 48 agents (24 layers x 2 kv heads), P in parallel. Resumable at agent level:
# agents whose run reached the final step are skipped; unfinished run dirs are moved aside (not deleted) and retrained.
# usage: scripts/train_ft_all.sh [config]   (default configs/train_0_5b_ft.yaml; control: configs/train_0_5b_ftctl.yaml)
set -euo pipefail
CFG=${1:-configs/train_0_5b_ft.yaml}
P=4
NAME=$(basename "$CFG" .yaml)
cd /workspace/kv/ml-learning-to-evict
source .env
[ -f "$CFG" ] || { echo "no config $CFG" >&2; exit 1; }
SWEEP=$(grep -m1 '^sweep_name:' "$CFG" | sed 's/^sweep_name: *"\{0,1\}\([^"]*\)"\{0,1\} *$/\1/')
STEPS=$(grep -m1 'num_epochs_or_steps:' "$CFG" | awk '{print $2}' | tr -d _)
[ -n "$SWEEP" ] && [ -n "$STEPS" ] || { echo "could not read sweep_name / num_epochs_or_steps from $CFG" >&2; exit 1; }
FINAL=$(printf '%012d.pth' "$STEPS")
LOG=/workspace/kv/logs/$NAME
mkdir -p "$LOG"
rm -f "$LOG/FAILED.txt"

TODO=()
for L in $(seq 0 23); do for H in 0 1; do
  d=agents/grouped/$SWEEP/$(printf 'layer_%06d/kv_head_%03d' "$L" "$H")
  done_runs=0
  if [ -d "$d" ]; then
    for run in "$d"/*/; do
      [ -d "$run" ] || continue
      if [ -f "$run/checkpoints/$FINAL" ]; then
        done_runs=$((done_runs + 1))
      else
        mkdir -p "agents/grouped/${SWEEP}_incomplete"
        mv "$run" "agents/grouped/${SWEEP}_incomplete/L${L}_H${H}_$(basename "$run")_$(date +%s)"
        echo "moved unfinished run $run aside"
      fi
    done
  fi
  [ "$done_runs" -le 1 ] || { echo "$d has $done_runs finished runs; resolve by hand" >&2; exit 1; }
  [ "$done_runs" -eq 1 ] || TODO+=("$L $H")
done; done
echo "start $(date '+%F %T') P=$P CFG=$CFG sweep=$SWEEP steps=$STEPS agents to train: ${#TODO[@]}"
if [ "${#TODO[@]}" -gt 0 ]; then
  printf '%s\n' "${TODO[@]}" | \
    xargs -P "$P" -n 2 sh -c 'uv run python src/kvcompression/entrypoints/rl/train_warmstart.py \
      --config '"$CFG"' target_layer_idx=$0 kv_head_idx=$1 > '"$LOG"'/L$0_H$1.log 2>&1 \
      || echo "FAILED L$0 H$1" >> '"$LOG"'/FAILED.txt'
fi
echo "end $(date '+%F %T')"
if [ -f "$LOG/FAILED.txt" ]; then cat "$LOG/FAILED.txt"; exit 1; fi
