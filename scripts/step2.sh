#!/usr/bin/env bash
set -euo pipefail
cd /workspace/kv
uv run --project ml-learning-to-evict python equiv_test.py 2>&1 | tee equiv_test_v3.log
uv run --project ml-learning-to-evict python evict_eval.py --method full --limit 1 --ks 8 --out_dir /workspace/kv/logs_smoke_v3 2>&1 | tee logs/smoke_v3_full.log
uv run --project ml-learning-to-evict python evict_eval.py --method streaming --budget 128 --limit 1 --ks 8 --out_dir /workspace/kv/logs_smoke_v3 2>&1 | tee logs/smoke_v3_streaming.log
echo STEP2 DONE
