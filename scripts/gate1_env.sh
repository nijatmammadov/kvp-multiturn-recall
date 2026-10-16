#!/usr/bin/env bash
set -euo pipefail
cd /workspace/kv
/venv/main/bin/pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
/venv/main/bin/pip install transformers datasets accelerate
/venv/main/bin/python -c "import torch, transformers, datasets; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), transformers.__version__, datasets.__version__)"
echo "gate1 env OK"
