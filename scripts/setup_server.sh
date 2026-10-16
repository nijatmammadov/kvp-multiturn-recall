#!/usr/bin/env bash
# One-shot setup of a fresh GPU instance (eval batches and the v4 warm-start fine-tune).
# usage: scripts/setup_server.sh /path/to/agents.tar.gz
# Expects this repo cloned at /workspace/kv/kvp-eviction. Builds Apple's repo + env next to it,
# downloads Qwen2.5-0.5B-Instruct, and installs the 48 trained agents.
set -euo pipefail

AGENTS_TAR=${1:?usage: setup_server.sh /path/to/agents.tar.gz}
REPO=$(cd "$(dirname "$0")/.." && pwd)
W=/workspace/kv
P=$W/ml-learning-to-evict
A=$P/agents/grouped/oasst2_0_5b

command -v uv >/dev/null || pip install uv
mkdir -p "$W"
[ -d "$P" ] || git clone https://github.com/apple/ml-learning-to-evict.git "$P"
git -C "$P" checkout d33c2dc
cp "$REPO/repo_changes/oasst2_dataset.py" "$REPO/repo_changes/recall_dataset.py" \
   "$REPO/repo_changes/mixed_qkv_dataset.py" "$P/src/kvcompression/data/"
cp "$REPO/repo_changes/train_warmstart.py" "$P/src/kvcompression/entrypoints/rl/"
cp "$REPO/repo_changes/preprocess_oasst2_0_5b.yaml" "$REPO/repo_changes/train_0_5b.yaml" \
   "$REPO/repo_changes/preprocess_recall_0_5b.yaml" "$REPO/repo_changes/train_0_5b_ft.yaml" \
   "$REPO/repo_changes/train_0_5b_ftctl.yaml" "$P/configs/"

cd "$P"
if [ ! -f .env ]; then
  sed 's#^export KVCOMPRESSION_MODEL_ID=.*#export KVCOMPRESSION_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct#' .env.template > .env
  echo "export AWS_EC2_METADATA_DISABLED=true" >> .env
fi
uv sync
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(torch.__version__, torch.cuda.get_device_name(0))"
uv run python -c "import transformers; assert transformers.__version__ == '4.55.4', transformers.__version__; print('transformers', transformers.__version__)"
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='models/Qwen2.5-0.5B-Instruct')"

TMP=$(mktemp -d)
tar xzf "$AGENTS_TAR" -C "$TMP"
SRC=$(dirname "$(find "$TMP" -type d -name layer_000000 | head -n1)")
[ -n "$SRC" ] && [ -d "$SRC/layer_000000" ] || { echo "no layer_000000 directory inside $AGENTS_TAR" >&2; exit 1; }
mkdir -p "$A"
cp -r "$SRC"/layer_* "$A"/
N=$(find "$A" -name best_ckpt.pth | wc -l)
[ "$N" -eq 48 ] || { echo "expected 48 best_ckpt.pth under $A, found $N" >&2; exit 1; }
rm -rf "$TMP"
echo "setup OK: 48 agents in $A"
