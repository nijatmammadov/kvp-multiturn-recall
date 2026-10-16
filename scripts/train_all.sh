#!/usr/bin/env bash
# Launch all 48 agents (24 layers x 2 kv heads), P in parallel.
# `uv run tune run` is replaced by `uv run python <recipe>` because the tune CLI
# crashes on import in this env (kagglehub/kagglesdk); single-device equivalent.
set -euo pipefail
P=4
STEPS=4000
cd /workspace/kv/ml-learning-to-evict
source .env
mkdir -p /workspace/kv/logs/train
echo "start $(date '+%F %T') P=$P STEPS=$STEPS"
for L in $(seq 0 23); do for H in 0 1; do echo "$L $H"; done; done | \
  xargs -P "$P" -n 2 sh -c 'uv run python src/kvcompression/entrypoints/rl/train_agent_sampler_distributed.py \
    --config configs/train_0_5b.yaml target_layer_idx=$0 kv_head_idx=$1 training.num_epochs_or_steps='"$STEPS"' \
    > /workspace/kv/logs/train/L$0_H$1.log 2>&1 || echo "FAILED L$0 H$1" >> /workspace/kv/logs/train/FAILED.txt'
echo "end $(date '+%F %T')"
