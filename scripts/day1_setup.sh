#!/usr/bin/env bash
set -euo pipefail

WORK=/workspace/kv
REPO_URL=https://github.com/apple/ml-learning-to-evict.git
COMMIT=d33c2dc

cd "$WORK"
[ -d ml-learning-to-evict ] || git clone "$REPO_URL"
cd ml-learning-to-evict
git checkout "$COMMIT"
echo "== HEAD: $(git rev-parse HEAD)"

echo "== torch pin in uv.lock"
grep -A1 '^name = "torch"$' uv.lock || echo "no torch entry in uv.lock"

echo "== uv sync"
uv sync

echo "== repo env check"
uv run python - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available in repo env"
print("device", torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok", (x @ x).float().abs().mean().item())
import torchtune
print("torchtune", torchtune.__version__)
try:
    import flash_attn
    print("flash_attn", flash_attn.__version__)
except ImportError:
    print("flash_attn: NOT INSTALLED in repo env")
EOF

echo "day1 setup OK"
