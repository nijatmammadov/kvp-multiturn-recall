# RESULTS — KVP multi-turn eval

## Gate 1 (2026-10-09)

### Repo setup
Command: `cd /workspace/kv && ./day1_setup.sh 2>&1 | tee setup.log`
- HEAD d33c2dcf8d8c2197ac92a2ceaf769c7f9b361b0d; uv.lock torch pin 2.10.0 (+cu128 variant); uv sync PASSED (Python 3.11.16).
- Repo env: torch 2.10.0+cu128, CUDA 12.8, available True, RTX 4090, bf16 matmul OK; torchtune 0.6.1; transformers 4.55.4.
- flash-attn 2.8.3 installed but import FAILS: `ModuleNotFoundError: No module named flash_attn_2_cuda` — pyproject sets `FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE`, so no CUDA kernels were built.
- torchao 0.15.0 warns: skipping cpp extensions (incompatible with torch 2.10.0).

### flash-attn audit (left broken on purpose)
Command: `grep -rn "flash_attn" --include=*.py .` in ml-learning-to-evict (plus `git grep -i flash`).
- No `.py` file in the repo (src/, scripts/, tests/) imports `flash_attn`. "flash" appears only in pyproject.toml (dependency + `FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE`).
- Trace collection (`unroll_and_store.py`): torchtune model (`qwen2_5_*` builder); attention runs via torchtune `MultiHeadAttention._attention_call = _sdpa_or_flex_attention()` (PyTorch SDPA / flex_attention, not the flash_attn package). `ExtractAttentionTensors` wraps that call and stores q/k/v to safetensors.
- Oracle reward (`rl/rewards/oracle.py`) + compression strategy: attention weights are computed explicitly from q,k by `attention_utils.materialize_attention_weights` (manual softmax), so no fused kernel needs to return weights.
- Agent training / compressible cache: `kv_cache/compressible_attention.py` uses `torch.nn.attention.flex_attention` (compiled).
- Conclusion: flash_attn is unused on trace collection, agent training and oracle paths; flash-attn 2.8.3 left without CUDA kernels (import fails with `No module named flash_attn_2_cuda`).

### Gate 1 env (/venv/main)
Command: `cd /workspace/kv && ./gate1_env.sh 2>&1 | tee gate1_env.log` (pip install torch==2.10.0 --index-url .../whl/cu128; pip install transformers datasets accelerate)
- Result: `2.10.0+cu128 12.8 True 5.19.0 5.1.0` (torch, cuda, available, transformers, datasets). torch matches uv.lock pin 2.10.0+cu128.
- Note: transformers here is 5.19.0 vs 4.55.4 in the repo env.

### Dialogues
Command: `cd /workspace/kv && /venv/main/bin/python mt_niah.py 2>&1 | tee mt_niah.log`
- `wrote 180 dialogues to mt_niah.jsonl (filler pool 15353)`; `wc -l` = 180 (60 each for k = 2, 4, 8; seed 0).

### Gate 1 result: PASS (full cache, Qwen2.5-0.5B-Instruct, bf16, greedy)
Command: `cd /workspace/kv && time /venv/main/bin/python gate1.py 2>&1 | tee gate1.log` → `gate1_full.jsonl`
- needle sentence tokens = 23, value tokens L = 5
| k | acc | len_min | len_mean | len_max | >256 | >512 |
|---|---|---|---|---|---|---|
| 2 | 1.000 | 229 | 357 | 461 | 0.98 | 0.00 |
| 4 | 1.000 | 423 | 609 | 709 | 1.00 | 0.90 |
| 8 | 1.000 | 859 | 1093 | 1280 | 1.00 | 1.00 |
- `Gate 1 (k=8, >= 90%): 1.000  PASS -> stay on 0.5B`
- Wall time: real 29m50s (unbatched HF generate, one turn at a time).
- Failures: none (180/180 correct).
- FLAG: all k=2 dialogues are <= 461 tokens, so a budget of b=512 never forces eviction at k=2.

## Decisions (2026-10-09)
- **Gate 1 decision: stay on Qwen2.5-0.5B-Instruct.** Reason: full-cache k=8 accuracy 1.000 (60/60) >= 0.90 threshold.
- **Budget decision: b in {128, 256} replaces {256, 512}.** Reason: b=512 never evicts at k=2 (max dialogue length 461 tokens; 0.00 of k=2 dialogues exceed 512).

### Repo env datasets check
Command: `cd ml-learning-to-evict && uv run python -c "import datasets, transformers; print(datasets.__version__, transformers.__version__)"`
- Output: `4.5.0 4.55.4` -> datasets IS present in the repo env (4.5.0; /venv/main has 5.1.0). transformers 4.55.4 (/venv/main has 5.19.0).

### Needle position (Qwen2.5 chat template, /venv/main tokenizer)
Command: inline /venv/main/bin/python script applying `tok.apply_chat_template([system, needle-user], tokenize=False)` over all 180 rows of mt_niah.jsonl.
- System block (`<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n`) = 11 tokens (positions 0-10).
- System + needle user turn (through its `<|im_end|>\n`) = 39-41 tokens (39 for k2_000).
- Value (5 tokens) sits at positions 31-35 in k2_000; first value token at 31-33, last at <= 37 across all 180.
- So the 4 sink tokens (0-3) are inside the system block; the needle value starts ~27 tokens after the last sink.


## Traces: OASST2 multi-turn, Qwen2.5-0.5B-Instruct (2026-10-09)
Branch `oasst2-0.5b` (commit 0ce3940: oasst2_dataset.py + configs/preprocess_oasst2_0_5b.yaml). `.env` from template, model id 0.5B, `AWS_EC2_METADATA_DISABLED=true`.

### Deviation: torchtune CLI broken in repo env
- `uv run tune download ...` and `uv run tune run ...` both crash on import: `ImportError: cannot import name 'get_web_endpoint' from 'kagglesdk.kaggle_env'` (uv.lock: kagglehub 1.0.1, kagglesdk 0.1.24). Log: `trace_smoke_tunecli_fail.log`. No versions changed.
- Weights instead: `uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='models/Qwen2.5-0.5B-Instruct')"` (huggingface-hub 0.36.2) -> model.safetensors, vocab.json, merges.txt present; checkpoint_files unchanged.
- Recipes instead: `uv run python <recipe>.py --config ... key=val` (equivalent to single-device `tune run`, which only sets sys.argv and calls runpy on the recipe).

### Template check
Command: `uv run python /workspace/kv/check_template.py models/Qwen2.5-0.5B-Instruct` (log check_template.log)
- `MATCH: 58 tokens`: torchtune tokenize_messages == HF apply_chat_template(add_generation_prompt=True).

### Dataset smoke
Command: step-5 snippet (log dataset_smoke.log)
- 500 samples; prompt tokens min 200 / mean 420.4 / max 1107. No KeyError, no non-alternating thread.

### Trace smoke (2 samples)
Command: `time uv run python src/kvcompression/entrypoints/datagen/unroll_and_store.py --config configs/preprocess_oasst2_0_5b.yaml dataset.n_samples=2 output_dir=data/smoke` (log trace_smoke.log)
- 96 attention_tensors.safetensors (2 x 24 layers x 2 kv heads). k [L,64], v [L,64], q_group [7,L,64], bf16; L consistent across all 48 heads.
- sample 0: prompt 454 + 300 generated = L 754 (41 MB); sample 1: 502 + 300 = 802 (43 MB). Both hit max_new_tokens=300.
- ~14.0 s/sample (tqdm); wall 1m18s incl. model + OASST2 load. S3 upload fails with NoCredentialsError and keeps files.
- Each sample dir also has all_tokens.safetensors, prompt_ntokens.safetensors, all_text.txt, generated_text.txt.

### Training config
`configs/train_0_5b.yaml` (commit a210727) = train_single_head.yaml with: sweep_name oasst2_0_5b; distributed False; train/val dataset_path -> oasst2_mt/Qwen2.5-0.5B-Instruct/temperature_0.00; train randomize_prompt_len True, min_prompt_len 16; agent.head_dim 64.

### Training smoke check (not the probe; 20 steps on the 2 smoke samples)
Command: `time uv run python src/kvcompression/entrypoints/rl/train_agent_sampler_distributed.py --config configs/train_0_5b.yaml target_layer_idx=12 kv_head_idx=0 sweep_name=smoke_check training.num_epochs_or_steps=20 training.eval_interval=10 data.dataset.{train,val}.dataset_path=data/smoke/...` (log train_smoke.log; split files data/smoke/.../train.txt=sample_0000000, val.txt=sample_0000001)
- Runs end to end: first step 33 s (torch.compile), then ~5 steps/s, train episode_reward ~ -5.3; wall 1m50s. Agent saved under agents/grouped/smoke_check/.
- eval_reward=0.000, eval_length=0.000 is an artifact: loaders use drop_last=True and val had 1 sample < eval batch 16. With 50 val samples, 48 are evaluated per eval (3 x 16).
- Ran concurrently with the full trace job.

### Full traces (500 samples)
Command: `time uv run python src/kvcompression/entrypoints/datagen/unroll_and_store.py --config configs/preprocess_oasst2_0_5b.yaml` (log traces.log; run in tmux window traces)
- 500 sample dirs, 24000 attention_tensors.safetensors (500 x 24 layers x 2 kv heads). No real errors; every sample logs the expected S3 NoCredentialsError and keeps files.
- Wall time 94m35s (tqdm 1:33:44, avg 11.25 s/sample; range ~8.5-19 s/sample; slowdown around samples 400-450 coincided with host load average ~30 from other tenants, GPU memory flat).
- L (prompt + generated) min/mean/max = 223 / 623 / 1357; generated min/mean/max = 9 / 203 / 300; 207/500 hit max_new_tokens=300.
- Disk: 17G for the trace dir; / at 35G used, 116G free.
- Splits (in trace dir): train.txt sample_0000000-0000399 (400), val.txt 0000400-0000449 (50), test.txt 0000450-0000499 (50, held out for Gate 2: never train/tune on it).
- GPU memory of the trace process grew slowly from ~1.5 GB to 10.6 GB over the run (did not affect results).

### Probe: one agent, L12 H0, 4000 steps
Command: `time uv run python src/kvcompression/entrypoints/rl/train_agent_sampler_distributed.py --config configs/train_0_5b.yaml target_layer_idx=12 kv_head_idx=0 sweep_name=probe` (log probe.log; GPU log probe_gpu.csv via `nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 30`)
- Wall time 16m06s (tqdm 4000 steps in 14:38, ~4.55 it/s; rest = startup/compile/final eval). GPU had the probe to itself.
- Peak GPU memory 4078 MiB. GPU utilization (30 s samples while training) mean ~45%, range 13-73%.
- Only errors: expected S3 upload failure at the end. Agent saved: agents/grouped/probe/layer_000012/kv_head_000/2026-10-09_21-28-10/best_ckpt.pth. final_reward -1.4702, final_length 410.04.
- eval_reward every 200 steps (eval_length 410.042 every time = 48 val samples):

| step | 200 | 400 | 600 | 800 | 1000 | 1200 | 1400 | 1600 | 1800 | 2000 |
|---|---|---|---|---|---|---|---|---|---|---|
| eval_reward | -2.897 | -2.264 | -1.819 | -1.640 | -1.566 | -1.528 | -1.506 | -1.493 | -1.484 | -1.478 |

| step | 2200 | 2400 | 2600 | 2800 | 3000 | 3200 | 3400 | 3600 | 3800 | 4000 |
|---|---|---|---|---|---|---|---|---|---|---|
| eval_reward | -1.475 | -1.473 | -1.472 | -1.471 | -1.471 | -1.470 | -1.470 | -1.470 | -1.470 | -1.470 |

- Plateau: within 0.01 of the final value from step 2000, within 0.005 from step 2200, flat to 3 decimals from step 3200.

## 48-agent training launch
- P = 4: 6 x 4.1 GB = 24.5 GB exceeds ~20 GB; 4 x 4.1 GB = 16.3 GB.
- Projection: 48 x 16.1 min / (4 x 0.7) = 276 min = 4.6 h <= 6 h budget, so STEPS = 4000 (full schedule, no truncation).
- Command: `/workspace/kv/train_all.sh` in tmux window `train` (log /workspace/kv/logs/train_all.log; per-agent logs /workspace/kv/logs/train/L{l}_H{h}.log; failures to /workspace/kv/logs/train/FAILED.txt). Same xargs loop as specified, with `uv run python <recipe>` instead of `uv run tune run` (tune CLI broken, see above).
- Note: the heuristic grid shares the GPU with training, so grid timings are not valid results.

## Equivalence test (fp32): CRASHED in C -> no grid runs
Command: `cd /workspace/kv && uv run --project ml-learning-to-evict python equiv_test.py 2>&1 | tee equiv_test.log` (run while 4 training agents share the GPU)
- D gather (CPU): PASS
- A full-cache incremental vs one-shot forward: max logit diff 8.34e-05, argmax flips 0 (PASS, < 1e-3)
- B explicit 4D causal mask vs default mask: max logit diff 0.00e+00 (PASS, < 1e-4)
- C: crashed on the first k=8 dialogue before any comparison: `ValueError: all input arrays must have the same shape` at evict_eval.py:187 `np.stack(kept)`.
  Cause: log_event appends `kept` (shape [24, 2, cache_len]) at every event, but before the cache first exceeds b the cache holds fewer than b rows, so early events have width < b and the arrays cannot be stacked. Same crash would hit every streaming/h2o grid run. Not patched; awaiting decision.

### Training: kept at 4,000 steps (user decision, 2026-10-09 ~22:00)
- Measured: 4 parallel agents at ~1.9-2.3 it/s each vs 4.55 alone (efficiency ~0.43; host load avg ~37 from other tenants) -> projected ~36.5 min/agent x 12 waves = ~7.3 h, exceeding the 6 h cap by ~1.3 h. User chose to keep 4,000 steps.
- An attempted stop (`pkill -f train_all.sh ...`) only killed the train_all.sh wrapper shell (and the remote shell that issued it); xargs (reparented to PID 1), its 4 sh -c wrappers and 4 trainers kept running. Verified at 21:59: L0_H0/L0_H1/L1_H0/L1_H1 at ~1030/4000 steps, no FAILED.txt. Run left untouched; no relaunch.
- Consequence: train_all.log will not get its final "end" line; use per-agent logs and FAILED.txt for completion.

### evict_eval.py logging fix (user-provided lines)
- In run_dialogue -> log_event, `kept` now padded to width b with -1 and raises if an entry exceeds b (see `diff -u evict_eval.py.orig evict_eval.py`). First equiv run log kept as equiv_test_run1_crash.log.

## Equivalence test (fp32), re-run: ALL PASS
Command: `cd /workspace/kv && uv run --project ml-learning-to-evict python equiv_test.py 2>&1 | tee equiv_test.log` (GPU shared with 4 training agents)
- D gather (CPU): PASS
- A full-cache incremental vs one-shot forward: max logit diff 8.34e-05, argmax flips 0 (threshold 1e-3)
- B explicit 4D causal mask vs default mask: max logit diff 0.00e+00 (threshold 1e-4)
- C k8_000: streaming b=128, len 914, masked (q,k) pairs 284045, max logit diff 8.51e-05, flips 0
- C k8_001: streaming b=128, len 1040, masked pairs 381880, max logit diff 7.25e-05, flips 0
- C k8_002: streaming b=128, len 1120, masked pairs 454080, max logit diff 7.15e-05, flips 0 (threshold 1e-2)

## Heuristic grid (bf16, sequential, GPU shared with training: timings NOT valid)
Command: `/workspace/kv/grid.sh 2>&1 | tee logs/grid_all.log` in tmux window grid = smoke (`evict_eval.py --method h2o --budget 128 --limit 2 --out_dir /workspace/kv/logs_smoke`) then, sequentially, `evict_eval.py --method $m --budget $b` for full 0, streaming 128, streaming 256, h2o 128, h2o 256 (per-run logs logs/grid_<m>_b<b>.log; results logs/<m>_b<b>_bf16.jsonl + .npz dirs).
### Smoke: h2o b=128, 2 dialogues per k (logs_smoke/)
- k=2: acc 0.500, needle survival at answer 0.006; k=4: acc 0.000, 0.002; k=8: acc 0.000, 0.001.
- k2_000 correct (77013); others answered 761 / 2558 / 23 / 12345 / 1. npz shapes OK: events (7,), needle (7,24,2), sys (7,24,2), kept (7,24,2,128). Peak mem ~0.97 GB.
- Training slowed to ~1.4-1.7 it/s per agent while eval runs alongside (was ~2.0).

### evict_eval.py: stream saved in npz (2026-10-09 22:13)
- Added `stream=np.array(res["stream"], dtype=np.int32)` to np.savez_compressed in main (diff vs evict_eval.py.v2_keptfix). Applied while `full b0` was running (started 22:10:33 on the old script): full_b0 npz files and logs_smoke/ have NO stream. All streaming/h2o grid runs start after the edit and include stream.

### Turn-1 replies, h2o b=128 k=2 smoke dialogues
Command: `uv run --project ml-learning-to-evict python evict_eval.py --method h2o --budget 128 --ks 2 --limit 2 --out_dir /workspace/kv/logs_smoke_stream` (re-run with stream; final answers identical to logs_smoke: 77013, 761), then `uv run --project ml-learning-to-evict python turn1_replies.py` (log logs/smoke_turn1_replies.log)
- k2_000 (value 77013, CORRECT): turn-1 reply "Im sorry, but I dont have any information about a copper chapel with the code 77013. Could you please provide more context or clarify your question?" -> REPEATS the code (also in final reply).
- k2_001 (value 76150, wrong: 761): turn-1 reply "Im sorry, but I dont have any information about a hidden mill M4. Could you please provide more context or clarify your question?" -> no code.
- So the one correct h2o answer came with the code echoed in the assistant turn-1 reply; `needle_at_answer` only tracks the user-turn needle positions, not echoed copies.
- Full-cache Gate 1 replies (gate1_full.jsonl, HF generate): turn-1 reply contains the code in 3/60 (k=2), 3/60 (k=4), 9/60 (k=8) dialogues.

- Overnight status logger: /workspace/kv/status_loop.sh in tmux window status, appends to overnight_status.log every 15 min (read-only; no stops/restarts).

### Grid results (finished 04:42:15 on 2026-10-10; 180/180 per config; 0 errors in grid_all.log)
Commands: grid.sh (see above); tables: `uv run --project ml-learning-to-evict python grid_table.py` (logs/grid_table.txt) and `uv run --project ml-learning-to-evict python needle_survival_fixed.py` (logs/needle_survival_fixed.txt)

**BUG: needle_at_answer / needle_heads_full_at_answer / sys survival in the JSONL+npz are WRONG.** evict_eval.survival() computes isin(cache_rows, needle).mean() = fraction of cache rows that are needle tokens (full cache shows 0.016/0.010/0.005 instead of 1.0). Accuracy is unaffected. Corrected survival recomputed from npz kept arrays (fraction of the 5 needle-value tokens present per head at answer time). evict_eval.py not changed.

Accuracy (all valid):

| method | b | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| full | 0 | 1.000 | 1.000 | 1.000 |
| streaming | 128 | 0.000 | 0.000 | 0.000 |
| streaming | 256 | 0.400 | 0.033 | 0.000 |
| h2o | 128 | 0.050 | 0.083 | 0.017 |
| h2o | 256 | 0.417 | 0.100 | 0.083 |

Corrected needle survival at answer (accuracy / mean fraction of needle tokens kept over 48 heads / fraction of heads keeping all 5):

| method | b | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| streaming | 128 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| streaming | 256 | 0.400 / 0.370 / 0.367 | 0.033 / 0.017 / 0.017 | 0.000 / 0.000 / 0.000 |
| h2o | 128 | 0.050 / 0.134 / 0.026 | 0.083 / 0.069 / 0.028 | 0.017 / 0.039 / 0.025 |
| h2o | 256 | 0.417 / 0.674 / 0.366 | 0.100 / 0.223 / 0.060 | 0.083 / 0.083 / 0.030 |

Turn-1 echo of the code (from streams):
- streaming b=128 k=2: turn-1 echo 2/60; correct with echo 0/2, correct without echo 0/58
- streaming b=128 k=4: turn-1 echo 4/60; correct with echo 0/4, correct without echo 0/56
- streaming b=128 k=8: turn-1 echo 2/60; correct with echo 0/2, correct without echo 0/58
- streaming b=256 k=2: turn-1 echo 2/60; correct with echo 1/2, correct without echo 23/58
- streaming b=256 k=4: turn-1 echo 4/60; correct with echo 1/4, correct without echo 1/56
- streaming b=256 k=8: turn-1 echo 2/60; correct with echo 0/2, correct without echo 0/58
- h2o b=128 k=2: turn-1 echo 2/60; correct with echo 2/2, correct without echo 1/58
- h2o b=128 k=4: turn-1 echo 4/60; correct with echo 4/4, correct without echo 1/56
- h2o b=128 k=8: turn-1 echo 2/60; correct with echo 0/2, correct without echo 1/58
- h2o b=256 k=2: turn-1 echo 2/60; correct with echo 2/2, correct without echo 23/58
- h2o b=256 k=4: turn-1 echo 4/60; correct with echo 4/4, correct without echo 2/56
- h2o b=256 k=8: turn-1 echo 2/60; correct with echo 2/2, correct without echo 3/58


## Saturday 2026-10-10: survival fix, analysis, KVP integration
### Files (v3)
- evict_eval.py backed up as evict_eval.py.v2_stream; new evict_eval.py (user-provided) = per-head survival() fix + KVP method (best_ckpt.pth EMA weights, original positions) + --agents_dir. `diff evict_eval.py.v2_stream evict_eval.py` shows only these changes. Added key_match.py, gate2.py, analyze.py (user-provided).
- NOTE: survival fields (needle_at_answer, needle_heads_full_at_answer, sys) in the existing streaming/h2o grid JSONL/npz files (logs/*_bf16.jsonl from the first grid) remain WRONG; analyze.py recomputes survival from saved kept arrays and ignores them. (My earlier proposed fix, swapping isin args, would also have been wrong: it pools both heads.)
### Regression checks (step2.sh, log logs/step2.log)
- `uv run --project ml-learning-to-evict python equiv_test.py` (equiv_test_v3.log): D PASS; A 8.34e-05, 0 flips; B 0.00e+00; C k8_000/001/002 8.51e-05/7.25e-05/7.15e-05, 0 flips; ALL PASS.
- full --limit 1 --ks 8 (logs_smoke_v3): k8_000 correct, needle_at_answer 1.0, needle_heads_full_at_answer 1.0.
- streaming b=128 --limit 1 --ks 8: k8_000 wrong (answer 123456), needle_at_answer 0.0 = corrected recomputation (0.000 for all streaming b128).
### analyze.py on heuristic grid (log analysis.log; outputs analysis/)
- summary.md, summary.csv, accuracy.csv, head_importance.csv, 4 x head_needle_survival_*_k8.csv written.
- CRASHED in plot(): `ValueError: yerr must not contain negative values` -> accuracy_vs_k.png NOT written. Likely float rounding of Wilson bounds at acc 0/1 (tiny negative error bar). Not patched; awaiting decision.
- n_echo in analyze.py counts any copy of the code before the final question (any assistant reply), so it is >= the turn-1-only echo counts reported earlier.

### analyze.py v2 (user edits): plot yerr clipped at 0; base rates added
- analyze.py.v1 kept; first outputs kept as analysis_run1_plotcrash/ (+ analysis_run1_plotcrash.log).
- Changes: `err = [[max(0.0, acc - ci_lo)], [max(0.0, ci_hi - acc)]]`; analyze_dialogue adds `n_seen = int(kept.max()) + 1`, `base = {c: share of category among positions < n_seen}`; summary rows gain base_<cat>.
- Command: `uv run --project ml-learning-to-evict python analyze.py 2>&1 | tee analysis.log` -> analysis/summary.md, summary.csv, accuracy.csv, head_importance.csv, head_needle_survival_*_k8.csv, accuracy_vs_k.png (written OK).
- kept vs base table: `python kept_vs_base.py` -> analysis/kept_vs_base.md (cells = kept share / base share).

| method | b | k | template | system | needle_turn | filler_user | question | assistant |
|---|---|---|---|---|---|---|---|---|
| h2o | 128 | 2 | 0.175 / 0.140 | 0.022 / 0.020 | 0.106 / 0.075 | 0.177 / 0.150 | 0.100 / 0.059 | 0.420 / 0.556 |
| h2o | 128 | 4 | 0.179 / 0.119 | 0.019 / 0.011 | 0.074 / 0.044 | 0.221 / 0.195 | 0.090 / 0.034 | 0.415 / 0.596 |
| h2o | 128 | 8 | 0.195 / 0.109 | 0.017 / 0.006 | 0.048 / 0.025 | 0.244 / 0.205 | 0.087 / 0.019 | 0.410 / 0.636 |
| h2o | 256 | 2 | 0.144 / 0.140 | 0.019 / 0.020 | 0.084 / 0.076 | 0.164 / 0.150 | 0.066 / 0.059 | 0.522 / 0.555 |
| h2o | 256 | 4 | 0.136 / 0.120 | 0.013 / 0.011 | 0.062 / 0.044 | 0.232 / 0.197 | 0.053 / 0.035 | 0.505 / 0.593 |
| h2o | 256 | 8 | 0.144 / 0.110 | 0.011 / 0.006 | 0.042 / 0.025 | 0.261 / 0.207 | 0.046 / 0.019 | 0.496 / 0.633 |
| streaming | 128 | 2 | 0.133 / 0.139 | 0.008 / 0.019 | 0.000 / 0.075 | 0.073 / 0.149 | 0.142 / 0.059 | 0.643 / 0.559 |
| streaming | 128 | 4 | 0.127 / 0.117 | 0.008 / 0.011 | 0.000 / 0.043 | 0.065 / 0.192 | 0.142 / 0.034 | 0.658 / 0.603 |
| streaming | 128 | 8 | 0.130 / 0.107 | 0.008 / 0.006 | 0.000 / 0.024 | 0.051 / 0.203 | 0.142 / 0.019 | 0.668 / 0.642 |
| streaming | 256 | 2 | 0.129 / 0.140 | 0.007 / 0.020 | 0.030 / 0.076 | 0.142 / 0.150 | 0.073 / 0.059 | 0.619 / 0.555 |
| streaming | 256 | 4 | 0.118 / 0.119 | 0.004 / 0.011 | 0.001 / 0.044 | 0.167 / 0.195 | 0.071 / 0.034 | 0.639 / 0.597 |
| streaming | 256 | 8 | 0.129 / 0.111 | 0.004 / 0.006 | 0.000 / 0.025 | 0.173 / 0.211 | 0.071 / 0.020 | 0.624 / 0.627 |

cells = kept share / base share (fraction of tokens seen so far)
### Step 4: training complete (2026-10-10 05:29)
- 48/48 per-agent logs with final_reward; logs/train/FAILED.txt absent; exactly one best_ckpt.pth under each of the 48 agents/grouped/oasst2_0_5b/layer_*/kv_head_*; all trainers + xargs exited by 05:34, GPU free. Only errors: expected S3 upload failure per agent.
- Run: 21:49:44 -> 05:28:57 (~7.65 h, P=4, 4000 steps; grid shared the GPU 22:07-04:42).

| layer | H0 final_reward | H1 final_reward |
|---|---|---|
| 0 | -1.3723 | -1.5334 |
| 1 | -1.4003 | -1.5932 |
| 2 | -1.5030 | -1.4693 |
| 3 | -1.9161 | -1.4060 |
| 4 | -2.2549 | -1.6269 |
| 5 | -1.7793 | -1.6280 |
| 6 | -1.4284 | -1.8295 |
| 7 | -2.5699 | -2.2213 |
| 8 | -2.0693 | -1.9906 |
| 9 | -2.2135 | -2.5661 |
| 10 | -1.6733 | -2.2124 |
| 11 | -2.5931 | -3.2639 |
| 12 | -1.4702 | -2.1941 |
| 13 | -2.6265 | -2.5934 |
| 14 | -2.4892 | -3.4916 |
| 15 | -2.0961 | -2.9202 |
| 16 | -3.0145 | -2.8501 |
| 17 | -1.6734 | -2.1850 |
| 18 | -2.0364 | -2.3594 |
| 19 | -2.1535 | -1.8005 |
| 20 | -1.9805 | -2.0267 |
| 21 | -1.7905 | -1.8733 |
| 22 | -1.6220 | -1.6022 |
| 23 | -1.5224 | -1.6970 |

### Step 5: key match -> KEY MATCH FAILED (stopped; no KVP evals run, Gate 2 not run)
Command: `uv run --project ml-learning-to-evict python key_match.py --trace_dir $D 2>&1 | tee key_match.log` (D = oasst2_mt trace dir; test samples 450, 465, 480, 495; HF bf16 eager vs torchtune traces, all 48 heads)
- Worst over 4 samples x 48 heads: k_rel 0.0226, v_rel 0.0604, min per-token key cosine 0.9831, kvp_scores vs agent.forward max abs diff 0.0, KVP top-k overlap HF vs trace 0.960, KVP rank correlation 0.9966.
- PASS kvp_scores == agent.forward; PASS keys rel err < 0.05; FAIL values rel err < 0.05 (0.060); FAIL min per-token key cosine > 0.99 (0.983); PASS top-k overlap > 0.9; PASS rank corr > 0.98.
- Rankings (what KVP actually uses) agree; the failures are on raw V magnitude error and the single worst-token key cosine. Which head/sample/token gives the worst values is not printed by the script.

### Key mismatch diagnosis: ordinary bf16 noise (PASS) -> KVP evals allowed
Command: `uv run --project ml-learning-to-evict python key_diag.py --trace_dir $D 2>&1 | tee key_diag.log` (pass rule fixed in the script before results: per layer trace-vs-HF rel error <= 2x the HF bf16-vs-fp32 noise floor for K and V; no head with worst V error > 3x the median)
- The original key_match.py raw-tensor thresholds (V rel err < 0.05, min per-token key cosine > 0.99) were guesses that turned out too strict: HF bf16 vs HF fp32 on the same tokens already shows V rel error up to 0.0609 (layer 4) and K up to 0.0226 (layer 1).
- Trace-vs-HF error matches the noise floor in every layer: worst ratio 1.21x (layer 0 V: 0.0057 vs 0.0047), typically 1.00-1.08x; 1st-percentile key cosine >= 0.9982 in every layer. Median worst-V rel error over heads 0.0328; outlier heads: none. Rule 1 PASS, rule 2 PASS.
- Downstream ranking agreement (key_match.py): KVP rank correlation HF vs trace >= 0.9966 and top-k overlap >= 0.96 on every head; kvp_scores == agent.forward exactly.
- Per-layer table (K trace / K floor / V trace / V floor / cos p1):
    layer  K trace  K floor  V trace  V floor  cos p1  ok
        0   0.0007   0.0022   0.0057   0.0047  1.0000  ok
        1   0.0210   0.0226   0.0393   0.0407  0.9982  ok
        2   0.0053   0.0053   0.0473   0.0485  0.9999  ok
        3   0.0180   0.0181   0.0475   0.0484  0.9985  ok
        4   0.0161   0.0160   0.0604   0.0609  0.9992  ok
        5   0.0194   0.0193   0.0532   0.0532  0.9988  ok
        6   0.0183   0.0180   0.0382   0.0379  0.9988  ok
        7   0.0153   0.0150   0.0328   0.0323  0.9993  ok
        8   0.0134   0.0132   0.0394   0.0389  0.9994  ok
        9   0.0226   0.0216   0.0331   0.0330  0.9987  ok
       10   0.0135   0.0133   0.0382   0.0373  0.9994  ok
       11   0.0217   0.0204   0.0294   0.0281  0.9989  ok
       12   0.0134   0.0130   0.0500   0.0461  0.9995  ok
       13   0.0140   0.0132   0.0346   0.0321  0.9993  ok
       14   0.0128   0.0123   0.0378   0.0363  0.9993  ok
       15   0.0152   0.0144   0.0297   0.0279  0.9991  ok
       16   0.0160   0.0155   0.0287   0.0277  0.9992  ok
       17   0.0121   0.0117   0.0285   0.0281  0.9992  ok
       18   0.0140   0.0135   0.0284   0.0275  0.9988  ok
       19   0.0120   0.0113   0.0327   0.0317  0.9992  ok
       20   0.0137   0.0129   0.0364   0.0349  0.9992  ok
       21   0.0134   0.0128   0.0379   0.0366  0.9989  ok
       22   0.0162   0.0166   0.0327   0.0326  0.9990  ok
       23   0.0152   0.0149   0.0337   0.0331  0.9992  ok

### Step 6: Gate 2 -> PASS
Command: `uv run --project ml-learning-to-evict python gate2.py --trace_dir $D 2>&1 | tee gate2.log` (50 test traces, repo Oracle.compute_cost, cost 1.0 = perfect, lower better; per-head CSV gate2_per_head.csv). Ran concurrently with key_diag.py.
- mean cost over heads: kvp 1.884, kvp_raw 2.277, streaming 2.421, h2o_prefix 1.895, random 10.221
- heads where KVP (protected) beats StreamingLLM: 41/48; beats H2O-prefix: 28/48; kvp_raw beats random: 48/48
- `Gate 2: PASS (KVP below StreamingLLM on most heads)`
- Sanity: random worst on all 48 heads; kvp_raw ~ -final eval_reward (e.g. L0H0 1.467 vs -1.372, L14H1 4.588 vs -3.492; test vs val split).
- KVP loses to StreamingLLM on 7 heads: L0H0, L4H0, L7H0, L7H1, L15H0, L16H1, L19H1 (by mean cost).

### Step 7: KVP on MT-NIAH
- Smoke (`--method kvp --budget 128 --limit 2 --out_dir logs_smoke_v3`): runs; 0/6 correct; k2 answers 77015 (true 77013) and 76152 (true 76150): near misses.
- Full runs: `kvp_grid.sh` (b=128 then b=256, `evict_eval.py --method kvp --budget $b`, logs logs/grid_kvp_b{128,256}.log), tmux window kvp, GPU otherwise idle so timings are valid.

### Step 7 results: KVP on MT-NIAH (GPU otherwise idle; timings valid)
- kvp b128 (05:49-06:33): k=2 acc 0.233, k=4 0.033, k=8 0.000; kvp b256 (06:33-07:09): k=2 0.867, k=4 0.333, k=8 0.000 (survival from fixed per-head survival()).
- mean seconds/dialogue (b, k, s): 128 2 7.2;128 4 13.1;128 8 24.2;256 2 7.2;256 4 11.2;256 8 16.8;

### Step 8: final analysis
Command: `uv run --project ml-learning-to-evict python analyze.py 2>&1 | tee analysis_final.log` -> analysis/ (summary.md, summary.csv, accuracy.csv, head_importance.csv, head_needle_survival_*_k8.csv, accuracy_vs_k.png, kept_vs_base.md). Previous heuristic-only outputs kept in analysis_heuristics_only/.


## Experiment batch v2

Eval-only batch v2, 2026-10-10, on a new vast.ai instance (RTX 4090 49 GB, driver 570.144 / CUDA 12.8; GPU unshared, so timings are valid). No training. All times UTC.

### Setup and regression checks
- Repo uploaded from the local clone at b67431f (the instance has no GitHub credentials); working tree restored from git to undo Windows CRLF line endings (`git rm -r --cached . && git reset --hard`). Agents uploaded as /workspace/agents.tar.gz (48 best_ckpt.pth in layer_*/kv_head_* layout, built from the local training export).
- `cd $R && scripts/setup_server.sh /workspace/agents.tar.gz 2>&1 | tee logs/setup_v2.log` -> `setup OK: 48 agents in /workspace/kv/ml-learning-to-evict/agents/grouped/oasst2_0_5b`; torch 2.10.0+cu128 CUDA OK; transformers 4.55.4. (~25 min; slow host network, ~4.5 MB/s.)
- `cd $R/eval && uv run --project $P python equiv_test.py $R/results/mt_niah.jsonl` (logs/equiv_test_v4.log): D PASS; A 8.34e-05, 0 flips; B 0.00e+00; C k8_000/001/002 8.51e-05 / 7.25e-05 / 7.15e-05, 0 flips; ALL PASS (identical to v1).
- Determinism vs v1: `uv run --project $P python eval/evict_eval.py --method kvp --budget 256 --ks 8 --limit 20 --data results/mt_niah.jsonl --agents_dir $A --out_dir results/logs_determinism` (logs/determinism_v2.log) -> **identical answers: 20/20** (>= 19 required).
- Smoke: `evict_eval.py --method kvp --pos_scale_to 600 --tag pos600` and `--method h2o_std`, b=128, --limit 2, out logs_smoke_v4: both complete; kvp_pos600 records carry `"pos_scale_to": 600`; k=2 answers equal v1 KVP (77015, 76152).

### Overlap check
`cd $P && uv run python $R/eval/overlap_check.py --model_dir models/Qwen2.5-0.5B-Instruct --data $R/results/mt_niah.jsonl --out $R/results/overlap_check.json` (logs/overlap_check.log)
- 28 of 821 unique MT-NIAH filler prompts appear in the 400 training dialogues (exact and whitespace-normalized match the same).
- Dialogues containing at least one such filler: k=2 6/60, k=4 6/60, k=8 15/60.

### The batch
`cd $R && scripts/exp_v2.sh 2>&1 | tee logs/exp_v2.log` (tmux window exp; results in results/logs_eval_v2/). Total wall time 149m35s.

| run | start | wall | k=2 | k=4 | k=8 |
|---|---|---|---|---|---|
| kvp_pos600 b=256 | 08:43:53 | 27.5 min | 52/60 | 20/60 | 0/60 |
| kvp_pos600 b=128 | 09:11:23 | 29.3 min | 14/60 | 2/60 | 0/60 |
| kvp_pos400 b=256 | 09:40:40 | 29.1 min | 53/60 | 11/60 | 1/60 |
| h2o_std b=256 | 10:09:43 | 31.9 min | 13/60 | 6/60 | 3/60 |
| h2o_std b=128 | 10:41:36 | 31.9 min | 3/60 | 3/60 | 1/60 |

### Analysis
- `python3 eval/paired_stats.py --logs results/logs_eval results/logs_eval_v2 --pairs kvp_pos600:kvp kvp_pos400:kvp kvp:h2o_std h2o_std:h2o kvp_pos600:h2o_std --out results/paired_stats_v2.md` (27 exact McNemar tests, Holm-adjusted).
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --project $P python eval/analyze.py --logs results/logs_eval_v2 --data results/mt_niah.jsonl --out results/analysis_v2` (logs/analysis_v2.log). The first attempt without the offline flags failed because the host could not resolve huggingface.co (logs/analysis_v2_dnsfail.log); the tokenizer was already cached, so results are unaffected.

**k=2 sanity check: PASS.** kvp_pos600 vs kvp at k=2: 0 discordant pairs at both budgets (in fact 0 answer differences out of 60 at each budget).

**Position-control verdict (rule fixed in advance, read at b=256, k=8): position coverage is NOT the explanation.** kvp_pos600 = 0/60 (v1 kvp 0/60; McNemar 0 vs 0, p = 1), which is <= 2/60. This supports query-blindness (H1).
- kvp_pos600 vs kvp has 0 discordant pairs in all 6 (b, k) cells: rescaling changes 8-9 wrong answers at k=4 and 44-45 at k=8 into other wrong answers, but never changes correctness. Needle survival at answer is unchanged (b=256, k=8: 0.077 vs 0.078).
- Sensitivity T=400 (b=256): k=2 53/60, k=4 11/60, k=8 1/60. vs kvp: k=4 2 vs 11 discordant (p = 0.022, p_holm = 0.45), k=8 1 vs 0. Compressing positions further hurts k=4 (not significant after Holm) and does not rescue k=8.

**H2O-std vs v1 H2O vs KVP** (accuracy; paired tests from paired_stats_v2.md):

| b | k | h2o_std | v1 h2o | kvp | kvp vs h2o_std (only-kvp / only-h2o_std, p_holm) | h2o_std vs h2o (p_holm) |
|---|---|---|---|---|---|---|
| 256 | 2 | 0.217 | 0.417 | 0.867 | 39 / 0, 9.8e-11 | 0 / 12, 0.012 |
| 256 | 4 | 0.100 | 0.100 | 0.333 | 15 / 1, 0.012 | 0 / 0, 1 |
| 256 | 8 | 0.050 | 0.083 | 0.000 | 0 / 3, 1 | 0 / 2, 1 |
| 128 | 2 | 0.050 | 0.050 | 0.233 | 13 / 2, 0.16 | 0 / 0, 1 |
| 128 | 4 | 0.050 | 0.083 | 0.033 | 1 / 2, 1 | 0 / 2, 1 |
| 128 | 8 | 0.017 | 0.017 | 0.000 | 0 / 1, 1 | 1 / 1, 1 |

- The original H2O split (half the budget as recent window) is not a stronger baseline here: it is significantly worse than v1 H2O at b=256, k=2 (13 vs 25 correct) and no better elsewhere. KVP beats it significantly at b=256 for k=2 and k=4.
- H2O-std needle survival at answer (b=256): 0.465 / 0.106 / 0.042 for k=2/4/8; its few k>=4 successes come from echoed copies (accuracy without echo 0.000 at k=4 and k=8).

### Outputs
- results/logs_eval_v2/*.jsonl, results/paired_stats_v2.md, results/analysis_v2/, results/overlap_check.json, results/logs_determinism/*.jsonl, logs/*_v2.log, logs/equiv_test_v4.log, logs/overlap_check.log.
- Per-dialogue .npz files: /workspace/results_v2_npz.tar.gz (gitignored, downloaded separately).

## Experiment batch v4

Warm-start fine-tune of the 48 KVP agents on OASST2 replay + recall traces (batch v4), 2026-10-10. Instance: RTX 4090 24 GB, driver 595.71 (CUDA 13.2), 32 cores, 62 GB RAM, 150 GB disk, ~100 MB/s network. Times UTC.

### 1. Setup
- Repo uploaded from the local clone at 2bb8b5d; `git checkout -- .` to fix Windows line endings. Agents uploaded as /workspace/agents.tar.gz (48 best_ckpt.pth, layer_*/kv_head_* layout).
- `cd $R && scripts/setup_server.sh /workspace/agents.tar.gz 2>&1 | tee logs/setup_v4.log` -> `setup OK: 48 agents`; torch 2.10.0+cu128 CUDA OK; transformers 4.55.4; 1m18s.
- (Two earlier instances were abandoned before any work: one had a 32 GB disk, one had a failing network.)

### 2. Traces
- Smoke: `unroll_and_store.py --config configs/preprocess_recall_0_5b.yaml dataset.limit=2 output_dir=data/smoke_recall` -> 2 sample dirs x 48 attention_tensors.safetensors; generated_text contains the values (UUIDs; 7-digit number). 18 s.
- Recall traces (600, `preprocess_recall_0_5b.yaml`, logs/traces_recall_v4.log): 28m21s, 2.82 s/sample; 29 GB. OASST2 traces regenerated (500, `preprocess_oasst2_0_5b.yaml`, logs/traces_oasst2_v4.log): 90m56s (ran concurrently with recall); 17 GB. 24000 files. Only tracebacks are the expected S3 upload failure, one per sample.
- OASST2 splits: train 0-399, val 400-449, test 450-499.
- `check_recall_traces.py --config configs/preprocess_recall_0_5b.yaml --trace_dir $RC` (logs/check_recall_v4.log): **PASS** (all prompt lengths match; 600 prompts, tokens min/mean/max 411/897/1500). Model accuracy on the recall prompts: all 569/600 = 0.948; single 234/234 = 1.000; multikey 174/178 = 0.978; multivalue 80/92 = 0.870; multiquery 81/96 = 0.844. Splits written: train 500, val 50, test 50.
- Gate 2, original agents, regenerated OASST2 test traces: `gate2.py --trace_dir $O --agents_dir $A --out results/gate2_v4_base_oasst2.csv` (logs/gate2_v4_base_oasst2.log): mean cost_kvp 1.884 (v1 1.884, diff 0.0000); 48/48 heads within 0.02 (max diff 0.0000). **PASS**: the regenerated traces reproduce v1 exactly. Summary: kvp 1.884, kvp_raw 2.277, streaming 2.421, h2o_prefix 1.895, random 10.221.

### 3. Warm-start smoke
`train_warmstart.py --config configs/train_0_5b_ft.yaml target_layer_idx=12 kv_head_idx=0 sweep_name=smoke_ft training.num_epochs_or_steps=20 training.eval_interval=10` (logs/train_ft_smoke.log): `Warm start: loaded 21 tensors`; datasets 400 + 500 train, 50 + 50 val. **PASS**. Eval reward at step 10 = -1.678 (= step 20; EMA frozen before step 100, so this is the base L12H0 agent on the mixed val set). Only error: the expected S3 upload failure at the end. 2m01s.

### 4. Fine-tune (started 16:43, restarted 16:48 with P=3)
- **Deviation: parallelism reduced from P=4 to P=3.** With mixed batches padded to ~1,600 tokens, each trainer reserves 6.7 GB of GPU memory (vs 4.1 GB in the OASST2-only run). At P=4, two trainers were already at 6.7 GB (22.5/24 GB with the kvp_rerun eval alongside), and 4 x 6.7 + 1.6 GB would exceed 24 GB.
- The P=4 run was stopped by PID after ~4 min (xargs, then sh wrappers, then trainers; no FAILED.txt). Its 4 partial logs are in /workspace/kv/logs/train_0_5b_ft/aborted_p4/ and logs/train_ft_v4_aborted_p4.log; the resume logic in train_ft_all.sh moved the 4 unfinished run dirs to agents/grouped/oasst2_recall_ft_0_5b_incomplete/ (not deleted).
- Relaunched as `/workspace/kv/train_ft_all_p3.sh` (identical to scripts/train_ft_all.sh except `P=3`; copy saved as scripts/train_ft_all_p3.sh.used) -> logs/train_ft_v4.log. At P=3: 3 x 6.7 GB + eval 1.6 GB = 21.7 GB, stable; 2.4-2.5 it/s per agent.
- Fine-tune finished 19:49 (wall 180m19s for the P=3 run). Check: 48 best_ckpt.pth, 48 runs reached step 1500, no FAILED.txt -> **PASS**. Training ~2.4-2.5 it/s per agent, ~10.5 min per wave of 3.
- Export: `uv run --project $P python eval/export_ft_metrics.py --agents_dir $A_FT --logs_dir /workspace/kv/logs/train_0_5b_ft --steps 1500 --eval_interval 100 --out training_ft` (logs/export_ft_v4.log). The first attempt failed (`L0 H0: no final_reward in log`) because the rich logger wraps `final_reward:` and its value onto two lines. **Fix (eval/export_ft_metrics.py):** regex `final_reward:\s+(?:\S+\.py:\d+\s+)?(-?\d+\.\d+)`, which also matches the one-line form. The parsed values equal the checkpoints' best_eval_reward for all 48 agents (max diff 0.0). The partial output is kept on the instance as training_ft_partial_parsefail (not committed).
  - **Agents whose best checkpoint is the first eval (= original weights): 0/48.** Mean mixed-val eval reward first -> last: -3.7015 -> -3.4847. All 48 improved. Best step: 1500 for 35 agents, 1400 for 9, 1300 for 4.
- Defence evals ran alongside the fine-tune (`/workspace/kv/exp_v4_alongside.sh` = the lead-2 generation, kvp_rerun and `scripts/exp_v4.sh defence`; logs/exp_v4_defence.log):
  - `mt_niah.py --seed 2 --lead 2 --out results/mt_niah_lead2.jsonl`.
  - **Determinism:** kvp_rerun (original agents, b=256, seed-0) vs v1 kvp: **180/180 identical answers** (>= 178 required) -> **PASS**.
  - **Lead-2 full cache:** 60/60 at k = 2, 4, 8 (>= 95%) -> **PASS**.

### 5. Gate 2 (logs/gate2_v4_*.log, results/gate2_v4_*.csv)

| agents | traces | cost_kvp | kvp_raw | streaming | h2o_prefix | random | KVP beats streaming | KVP beats h2o_prefix |
|---|---|---|---|---|---|---|---|---|
| base | OASST2 test | 1.884 | 2.277 | 2.421 | 1.895 | 10.221 | 41/48 | 28/48 |
| ft | OASST2 test | 1.876 | 2.246 | 2.421 | 1.895 | 10.221 | 41/48 | 28/48 |
| base | recall test | 4.300 | 5.300 | 4.594 | 4.848 | 12.858 | 26/48 | 40/48 |
| ft | recall test | 4.023 | 4.906 | 4.594 | 4.848 | 12.858 | 28/48 | 41/48 |

- **Rule 3 (chat utility): preserved.** ft mean cost_kvp 1.876 <= 1.02 x 1.884 = 1.922 (slightly better). Recall ranking cost improves from 4.300 to 4.023.

### 6. Rank screen and MT-NIAH with the fine-tuned agents
- `eval/rank_screen.py --data results/mt_niah.jsonl --agents base=$A ft=$A_FT --budgets 128,256 --out results/rank_screen_v4.md` (29 min):

| agents | k | mean len | code pct | kept@128 | all5@128 | kept@256 | all5@256 |
|---|---|---|---|---|---|---|---|
| base | 2 | 317 | 0.476 | 0.326 | 0.108 | 0.801 | 0.605 |
| base | 4 | 540 | 0.407 | 0.092 | 0.021 | 0.335 | 0.120 |
| base | 8 | 959 | 0.341 | 0.021 | 0.001 | 0.080 | 0.019 |
| ft | 2 | 317 | 0.494 | 0.351 | 0.123 | 0.816 | 0.625 |
| ft | 4 | 540 | 0.425 | 0.110 | 0.022 | 0.363 | 0.142 |
| ft | 8 | 959 | 0.358 | 0.024 | 0.001 | 0.096 | 0.020 |

- Single-cut approximation vs measured v1 code survival (0.80 / 0.33 / 0.08): base kept@256 is 0.801 / 0.335 / 0.080, so the gap is < 0.01 at every k.
- **Rule 2 (mechanism):** the ft mean code percentile at k=8 (0.358) exceeds base (0.341): yes, but small. The measured ft code survival at b=256, k=8 (needle_at_answer) is 0.093, **below** the overall kept share (~0.27): the ft agents still rank the code below an average token at k=8.
- `scripts/exp_v4.sh ft` (logs/exp_v4_ft.log): kvp_ft b=256 22:12-22:43 (30.6 min, 10.2 s/dialogue); kvp_ft b=128 22:43-23:13. kvp_ft accuracy at b=256: 0.950 / 0.450 / 0.033; at b=128: 0.367 / 0.033 / 0.000 (k = 2/4/8).
- **Not run (out of credit, see step 8):** kvp_ft on lead-2 (b=256) and kvp_ft_pool4 (b=256).

### 7. Analysis
- `python3 eval/paired_stats.py --logs results/logs_eval results/logs_eval_v4 --pairs kvp_ft:kvp kvp_ft:h2o kvp_ft:streaming streaming_first64:kvp_ft streaming_first64:kvp --pool_ks 4,8 --out results/paired_stats_v4.md` (pair kvp_ft_pool4:kvp_ft dropped because that run is missing).
- `python3 eval/paired_stats.py --logs results/logs_eval_v4_lead2 --pairs kvp:h2o kvp:streaming streaming_first64:kvp --pool_ks 4,8 --out results/paired_stats_v4_lead2.md` (pairs with kvp_ft dropped because that run is missing).
- `python eval/partial_survival.py --logs results/logs_eval_v4 results/logs_eval_v4_lead2 --out results/fragmentation_v4.md`, run locally after the download. On the instance it crashed on the empty, unfinished lead-2 kvp_ft JSONL, which was removed.

**Rule 1, primary test (pre-registered: pooled k in {4, 8}, b=256, exact McNemar, unadjusted): kvp_ft vs kvp = 10 vs 1 discordant, p = 0.012 -> the recall-data warm start improves multi-turn retention.**
- The gain is mostly at k=4: 27/60 vs 20/60 (8 vs 1 discordant, p = 0.039, p_holm = 0.50). At k=8: 2/60 vs 0/60.
- At b=128 the pooled test is 0 vs 0 (p = 1). No per-cell test is significant after Holm.
- kvp_ft vs h2o (b=256, pooled k=4,8): 23 vs 5, p = 0.00091. kvp_ft vs streaming: 27 vs 0, p = 1.5e-8.

**Rule 5, benchmark shortcut: TRIGGERED.** streaming_first64 scores 60/60 at every k at both b=256 and b=128 on the seed-0 set (needle survival 1.000). **MT-NIAH as built (needle at the start of turn 1) is solved by keeping the first 64 tokens.** first64 beats kvp_ft and kvp in every pooled comparison (e.g. b=256: 91 vs 0 against kvp_ft). On the lead-2 set (two filler turns before the needle) the shortcut disappears:

| lead-2, b=256 | k=2 | k=4 | k=8 |
|---|---|---|---|
| full | 60/60 | 60/60 | 60/60 |
| kvp (base) | 14/60 | 3/60 | 0/60 |
| h2o | 3/60 | 9/60 | 2/60 |
| streaming | 19/60 | 1/60 | 0/60 |
| streaming_first64 | 7/60 | 0/60 | 0/60 |

On lead-2, base KVP beats H2O at k=2 (12 vs 1, p_holm = 0.031); nothing else is significant. KVP is not better than StreamingLLM there (7 vs 12 at k=2). kvp_ft on lead-2 was not run.

**Fragmentation** (results/fragmentation_v4.md): share of (dialogue, layer, head) pairs holding all 5 code tokens at b=256 is kvp_rerun 0.603 / 0.121 / 0.017 vs kvp_ft 0.621 / 0.142 / 0.017 (k = 2/4/8). Fragmented share (some but not all) for kvp_ft: 0.324 / 0.475 / 0.218.

### 8. Control: NOT RUN (stopped for lack of credit)
- Rule 1 was positive, so the control (OASST2-only warm start) was required. It started at 23:04 (`/workspace/kv/train_ft_all_p3.sh configs/train_0_5b_ftctl.yaml`, logs/train_ftctl_v4.log) and was stopped by PID about 1 min later, because the vast.ai balance ($0.52, about 1 h) could not cover its ~3.5 h. **Without the control, the gain cannot be attributed to the recall data rather than to extra training alone.** Partial logs: logs/train_v4_agents/train_0_5b_ftctl_aborted/.
- The eval queue was stopped after kvp_ft b=128 completed (step-6 script and its exp_v4.sh killed by PID; no eval process left).

### 9. Archives
- /workspace/results_v4.tar.gz (5.4 MB), /workspace/results_v4_npz.tar.gz (53 MB) and /workspace/agents_ft.tar.gz (442 MB, oasst2_recall_ft_0_5b with checkpoints) were downloaded to development/results_v4/ and md5-verified. Committed from the local clone (the instance has no GitHub credentials). Per-agent training logs are in logs/train_v4_agents/.
- /workspace/oasst2_traces.tar.gz (requested): **not created**, because 17 GB would not download within the remaining credit. Regenerate with `preprocess_oasst2_0_5b.yaml` (~1.5 h on a 4090).

### Wall time per step (UTC)
Setup 1m18s; traces 90m56s (recall 28m21s, in parallel); recall check ~5 min; Gate 2 base 2.5 min; warm-start smoke 2m01s; fine-tune 180m19s (plus ~4 min for the aborted P=4 start); Gate 2 x3 ~14 min; rank screen 29 min; kvp_ft b=256 30.6 min; kvp_ft b=128 ~30 min; defence evals 16:44-22:34 (alongside the fine-tune).

## Experiment batch v5

Eval-only batch v5 (including amendments 1 and 2), 2026-10-13. Word-needle set (invented pet name + city, lead 2) plus the open v4 digit items (kvp_ft on lead-2, score pooling). Instance: RTX 4090 24 GB, 256 cores, 472 GB RAM, 150 GB disk, ~100 MB/s from PyPI and Hugging Face. Times UTC.

### 1. Setup and regression checks
- Repo uploaded from the local clone at e57b4ec; `git checkout -- .` for line endings. `/workspace/agents.tar.gz` built on the server from the repo's `agents/oasst2_0_5b` (48 best_ckpt.pth). The instance has no GitHub credentials (it can neither fetch nor push), so later commits reached it as git bundles from the local clone (7eb4124, then 38968bd), and all results were committed and pushed from the local clone.
- `scripts/setup_server.sh /workspace/agents.tar.gz` (logs/setup_v5.log): **PASS**, `setup OK: 48 agents`; torch 2.10.0+cu128 CUDA OK; transformers 4.55.4.
- `test_score_pooling.py`: 5/5 **PASS**. `test_word_needle.py`: 6/6 **PASS** (logs/test_word_needle_v5.log; re-run after each amendment: logs/test_word_needle_v5_q2.log, logs/test_word_needle_v5_amend2.log, the last including `test_committed_sets_reproduce` on the q1 word set).
- `equiv_test.py results/mt_niah.jsonl` (logs/equiv_test_v5.log): **ALL PASS**; full-cache incremental vs one-shot max logit diff 8.3e-05, streaming masked-attention max diff 8.5e-05, 0 argmax flips.
- **Determinism:** `evict_eval.py --method kvp --budget 256 --ks 8 --limit 20` (logs/determinism_v5.log) vs v1 `kvp_b256_bf16`: **20/20 identical answers** (>= 19 required) -> **PASS**.

### 2. Word set and gate history
- Generated with `scripts/exp_v5.sh gen` (seed 4, lead 2): 180 dialogues, 60 per k; needle at user-turn index 2 in all, question in the last turn, needle text in no other turn; 100 distinct names, 20 cities. Smoke (`--method full --limit 2`, logs/smoke_v5.log): 6/6 correct, `name_heads_full_at_answer` = `city_heads_full_at_answer` = 1.0.
- **Gate (rule 1), question v1:** full cache 56/60, 59/60, 60/60 at k = 2/4/8 (needle survival 1.000 everywhere) -> **FAILED** (0.933 < 0.95 at k=2). 5 errors, all answer format: name read as a breed (2), previous filler task continued (2), title dropped ("Truffle") (1). Record: results/mt_niah_words_lead2_q1.jsonl, results/logs_eval_v5_words_q1/full_b0_bf16.jsonl (5fdffa6), logs/exp_v5_words_q1.log.
- **Amendment 1, question v2** ("Change of topic. Earlier I told you the full name of my {owner}'s {animal} ... separated by a comma."), same seed, only the question differs: full cache 48/60, 39/60, 37/60 -> **FAILED**, worse. 54 of 56 errors have the right city and a wrong name: the model named the owner and animal ("Teacher's Cat, Tampere") or invented a person ("Vincent van Gogh, Ghent"). Record: results/mt_niah_words_lead2_q2.jsonl, results/logs_eval_v5_words_q2/full_b0_bf16.jsonl (3b6f757), logs/exp_v5_words_q2.log.
- **Amendment 2:** back to question v1; the gate is replaced by conditioning every word-set statistic for rules 2a, 3, 4 and 6 on the 175/180 dialogues the full cache answered correctly (`--condition_on full:0`); unconditional numbers are secondary. No method had been run on any word set when either amendment was made. **The pre-registered gate failed twice; the word-set results below are conditional on full-cache correctness.**

### 3. Runs
Words and digits ran concurrently on the one GPU (as instructed), so per-config times are not comparable with v4's single-run times.
- `exp_v5.sh digits` (logs/exp_v5_digits.log), 14:57:54-16:46:44: kvp_ft lead-2 (23.7 min); kvp_ft_pool4 seed-0, lead-2; kvp_ft_pool8 seed-0, lead-2.
- `exp_v5.sh words` on the q1 set (logs/exp_v5_words.log), 15:34:06-19:30:57: the full-cache config was reused (all 180 ids present; printout 56/60, 59/60, 60/60), then kvp, h2o, streaming at b=256 and kvp_ft, kvp_ft_pool4, kvp_ft_pool8 at b=256 and b=128, ~25-27 min each. Exit 0, no errors.

Accuracy per k (b=256 unless noted; words conditioned on full:0, n = 56/59/60; unconditional words in results/paired_stats_v5_words_uncond.md; lead-2 kvp, h2o, streaming and seed-0 kvp_ft (b=256 and b=128) are from v4; empty = not run):

| config | words k=2 | words k=4 | words k=8 | digits lead-2 k=2 | k=4 | k=8 | digits seed-0 k=2 | k=4 | k=8 |
|---|---|---|---|---|---|---|---|---|---|
| kvp | 17/56 | 0/59 | 0/60 | 14/60 | 3/60 | 0/60 | | | |
| h2o | 40/56 | 37/59 | 23/60 | 3/60 | 9/60 | 2/60 | | | |
| streaming | 12/56 | 0/59 | 0/60 | 19/60 | 1/60 | 0/60 | | | |
| kvp_ft | 15/56 | 2/59 | 0/60 | 22/60 | 8/60 | 0/60 | 57/60 | 27/60 | 2/60 |
| kvp_ft_pool4 | 9/56 | 0/59 | 0/60 | 27/60 | 4/60 | 0/60 | 59/60 | 13/60 | 2/60 |
| kvp_ft_pool8 | 10/56 | 0/59 | 0/60 | 21/60 | 2/60 | 0/60 | 60/60 | 33/60 | 1/60 |
| kvp_ft, b=128 | 1/56 | 0/59 | 0/60 | | | | 22/60 | 2/60 | 0/60 |
| kvp_ft_pool4, b=128 | 0/56 | 0/59 | 0/60 | | | | | | |
| kvp_ft_pool8, b=128 | 1/56 | 0/59 | 0/60 | | | | | | |

`pool_changed` > 0 in every pooled record except 9 seed-0 k=2 dialogues per pooled config (pool4 and pool8, b=256), where it is None: these dialogues have total_len <= 258 (all others >= 263), never exceeded the budget at an eviction step (needle_at_answer 1.0), so pooling never ran. All word-set and lead-2 pooled records have `pool_changed` > 0 (min 0.010).

### 4. Analysis
Analysis commands (with the amendment-2 words commands): `paired_stats.py` -> results/paired_stats_v5_words.md (conditioned), results/paired_stats_v5_words_uncond.md, results/paired_stats_v5_digits_lead2.md, results/paired_stats_v5_digits_seed0.md; `partial_survival.py` -> results/fragmentation_v5.md. Per-config means of `needle_at_answer`, `needle_heads_full_at_answer`, `name_ok`, `city_ok`, `name_heads_full_at_answer` etc.: new script `eval/v5_means.py` -> results/means_v5_words.md (conditioned), results/means_v5_words_uncond.md, results/means_v5_digits_lead2.md, results/means_v5_digits_seed0.md. Rule-3 means over all k were computed from the same JSONLs.

**Rule 2, primary (pooling; kvp_ft_pool4 vs kvp_ft, b=256, k in {4, 8} pooled, exact McNemar, Holm over 2): NOT SUPPORTED on either needle type.**

| set | n | only pool4 right | only kvp_ft right | p | p_holm |
|---|---|---|---|---|---|
| (a) words, conditioned | 119 | 0 | 2 | 0.50 | 0.68 |
| (b) digits lead-2 | 120 | 3 | 7 | 0.34 | 0.68 |

Both differences point against pooling. Unconditional words: 0 vs 2 (n = 120), p = 0.50. pool8 (not part of the rule): words 0 vs 2, p = 0.50; digits lead-2 1 vs 7, p = 0.07.

**Rule 3, mechanism (b=256, pool4 vs kvp_ft): NOT MET on both sets. The pattern holds on words, not on digits.**

| set | needle_heads_full_at_answer kvp_ft -> pool4 | needle_at_answer kvp_ft -> pool4 | condition |
|---|---|---|---|
| words, conditioned, all k (n = 175) | 0.049 -> 0.085 (up) | 0.262 -> 0.228 (-0.035) | met |
| words, conditioned, k = 4, 8 (n = 119) | 0.020 -> 0.043 (up) | 0.176 -> 0.147 (-0.029) | met |
| digits lead-2, all k (n = 180) | 0.069 -> 0.078 (up) | 0.225 -> 0.266 (+0.041) | not met (> 0.02 rise) |
| digits lead-2, k = 4, 8 (n = 120) | 0.022 -> 0.026 (up) | 0.137 -> 0.168 (+0.032) | not met |

Read with rule 2: on words, pooling raises the share of heads holding the whole needle (name: 0.101 -> 0.138, city: 0.127 -> 0.299) while token survival falls, and accuracy does not rise (2/175 fewer at k = 4; 15/56 -> 9/56 at k = 2). This matches "whole-needle survival up but accuracy flat". On digits, pooling raised both whole-needle and token survival, and accuracy still did not rise (k = 4: 8/60 -> 4/60).

**Rule 4, budget prediction (words, b=128, kvp_ft_pool8 accuracy <= kvp_ft): MET**, but at the floor: 1/175 vs 1/175 (both at k=2; unconditional 1/180 vs 1/180); pool4 0/175. No pool8 gain at b=128, so nothing contradicts the free-slot explanation, but with kvp_ft at 0.006 the test has almost no room to show a gain.

**Rule 5, position invariance (digits, kvp_ft on lead-2, b=256): MET.** k=2: 22/60 = 0.367 (predicted [0.25, 0.50]); k=8: 0/60 = 0.000 (predicted <= 0.05). k=4: 8/60 = 0.133. No sign that position matters for the fine-tuned agents beyond what the ranges allow. Vs base kvp on lead-2 (not a rule): pooled k = 4, 8 5 vs 0, p = 0.062; k=2 10 vs 2, p = 0.039 (p_holm 0.46).

**Rule 6, words vs digits (descriptive): expectation NOT MET.** kvp_ft, b=256:

| k | words acc (conditioned) | words acc (unconditional) | digits lead-2 acc | words name_ok | words city_ok |
|---|---|---|---|---|---|
| 2 | 15/56 = 0.268 | 15/60 = 0.250 | 22/60 = 0.367 | 0.321 | 0.643 |
| 4 | 2/59 = 0.034 | 2/60 = 0.033 | 8/60 = 0.133 | 0.051 | 0.203 |
| 8 | 0/60 = 0.000 | 0/60 = 0.000 | 0/60 = 0.000 | 0.000 | 0.000 |

(name_ok, city_ok conditioned; unconditional 0.300/0.600, 0.050/0.200, 0/0.) Words are below digits at k = 2 and 4 and equal at k = 8, although more word-needle tokens survive (needle_at_answer 0.445/0.240/0.113 words vs 0.402/0.199/0.074 digits). The city is recalled about twice as often as the name; the name, not the city, limits word accuracy.

**Other comparisons on words (conditioned, b=256; paired_stats_v5_words.md):** kvp_ft vs h2o pooled k = 4, 8: 0 vs 58, p = 6.9e-18; H2O is far ahead on the word set (40/56, 37/59, 23/60), the reverse of the digit sets. kvp_ft vs kvp: 2 vs 0, p = 0.50; kvp_ft vs streaming: 2 vs 0, p = 0.50.

**Fragmentation** (results/fragmentation_v5.md, digits): share of (dialogue, layer, head) pairs holding all 5 code tokens at b=256, lead-2: kvp_ft 0.162 / 0.037 / 0.007, pool4 0.182 / 0.045 / 0.007, pool8 0.397 / 0.189 / 0.081 (k = 2/4/8). Fragmented share drops with pool8 (0.502 -> 0.232 at k=2).

### 5. Exploratory analyses (not pre-registered)
**Descriptive only, not part of any rule verdict.** Script `eval/v5_exploratory.py`, output results/v5_exploratory.md (run on the server without `--partial`).
- **A. Pooling on digits, b=256.** Pooling does not lower code survival; it raises it. lead-2 needle_at_answer kvp_ft -> pool4 -> pool8: k=2 0.402 -> 0.460 -> 0.546; k=4 0.199 -> 0.234 -> 0.289. needle_heads_full rises most with pool8 (k=2 0.162 -> 0.397). Accuracy does not follow: lead-2 k=4 8 -> 4 -> 2 of 60. On seed-0, pool4 loses at k=4 (only pool4 right 3, only kvp_ft right 17; 27 -> 13 of 60) while pool8 gains (13 vs 7; 27 -> 33). So the survival changes are a redistribution that does not translate into recall.
- **B. Words vs digits at matched length (150-token bins of total_len).** kvp_ft: words accuracy is at or below digits in every bin with data (e.g. 450-599: 0.240 vs 0.262; 600-749: 0.125 vs 0.194) although words keep more needle tokens (0.443 vs 0.357; 0.352 vs 0.242). Base kvp is the other way round (450-599: 0.286 vs 0.136). Name/city split (kvp_ft b=256, k=2): name only 0.054, city only 0.375, both 0.268, neither 0.304.
- **C. Echoes.** Full-cache replies to the needle turn repeat name and city in 85/175 word dialogues, one of them in 48, neither in 42; the code is repeated in 15/180 lead-2 digit dialogues. H2O depends strongly on the echo: words 71/85 correct with both echoed vs 1/42 with none; digits 11/15 with the code echoed vs 3/165 without. KVP and kvp_ft depend on it much less (words kvp_ft 15/85, 2/48, 0/42; digits 6/15 vs 24/165). This fits H2O keeping a second, later copy of the needle in the reply, which may explain much of H2O's advantage on words; it is not tested here.

### 6. Archives and commits
- Commits from the local clone: 5fdffa6 (q1 gate record), 3b6f757 (q2 gate record), 31e2a23 (digits part and setup/check logs), and the final v5 commit (words JSONLs, stats, means, exploratory, scripts, logs, this section). All files downloaded from the server were md5-checked against the server copies.
- `/workspace/results_v5_npz.tar.gz` (102 MB, 2906 .npz files: the 10 word-q1 configs, the word-q2 full-cache run, the 5 v5 digit configs, determinism and smoke) downloaded to development/results_v5/, md5 ba1768a2b2b4452c2240e9b4aa7b9c8c (matches the server). Not in git.
- The instance was left running (not destroyed).

### Wall time (UTC)
Setup and checks 14:18-14:28; word gate q1 14:28:49-14:53; gate q2 14:58:39-15:23; digits 14:57:54-16:46:44 (5 configs); words 15:34:06-19:30:57 (9 method configs, ~25-27 min each, concurrent with digits until 16:46); analysis 19:31-19:32.

## Experiment batch v6

Head-breadth test (batch v6), 2026-10-15. Instance: RTX 4090 48 GB, 112 cores, 503 GB RAM, 150 GB disk, ~11-12 MB/s from PyPI and Hugging Face. Times UTC.

### 1. Setup and checks
- Repo brought to the server as a git bundle of the local clone at 390474d (md5 acda5b0a7212b46046b838a1b403b02d on both sides), cloned to `/workspace/kv/kvp-eviction`. `/workspace/agents.tar.gz` built from the repo's `agents/oasst2_0_5b`.
- `scripts/setup_server.sh` (logs/setup_v6.log): **PASS**, `setup OK: 48 agents`; torch 2.10.0+cu128, transformers 4.55.4.
- `test_score_pooling.py`: 5/5 **PASS** (logs/test_score_pooling_v6.log). `test_v6.py`: 3/3 **PASS** (logs/test_v6.log).
- `exp_v6.sh check` (logs/check_v6.log), 06:12-06:18: `regression: 10/10 answers identical to v5 kvp_ft`; `smoke OK: heads48 holds the needle in all heads, heads6 in >= 6`. **PASS**. Smoke heads48 answers (k=4): 000 "Sir Pickles, Aarhus" correct; 001 truth Lady Waffle | Bergen, answer "Bella", New York (wrong); 002 "Madame Dumpling, Ghent" correct.

### 2. Runs
- `exp_v6.sh breadth` was started at 06:18:35 and stopped at 06:19:11 on the user's request, before n = 6 finished (1 dialogue written, kept: results/logs_eval_v6_breadth/kvp_ft_heads6_b256_bf16.jsonl; logs/exp_v6_breadth.log).
- On the user's request the n = 48 level was run first, alone (logs/exp_v6_n48.log), 06:19:23-06:54:12, exit 0:
  `uv run --project $P python eval/evict_eval.py --data results/mt_niah_words_lead2_q1.jsonl --agents_dir agents/oasst2_recall_ft_0_5b --out_dir results/logs_eval_v6_breadth --method kvp --budget 256 --protect_needle value --protect_heads 48 --tag ft_heads48 --ks 4,8`
  (relative `--agents_dir`, so the records' `agents_dir` field differs in spelling from the script's absolute path; same agents). All 120 records have `protect_heads` = 48 and `needle_heads_full_at_answer` = 1.0.
- The user's instruction: run n = 6, 12, 24 only if n = 48 accuracy (k = 4, 8, conditioned on full cache correct) is >= 0.80. It was not, so **n = 6, 12, 24 were not run** and `head_breadth.py` (which needs every level) was not run.

### 3. Result
**B2, manipulation check: NOT MET.** kvp_ft b=256, needle (name + city tokens) force-kept in all 48 heads, words q1, conditioned on full:0 correct (119 of 120; excluded w_lead2_k4_014):

| k | correct / n |
|---|---|
| 4 | 34/59 = 0.576 |
| 8 | 5/60 = 0.083 |
| 4 + 8 | **39/119 = 0.328** (needs >= 0.80) |

name_ok 39/119, city_ok 51/119 (vs v5 kvp_ft n = 0: 2/119 correct). Force-keeping the name and city tokens in every head does not restore recall, so **B1 cannot be read as a breadth effect and was not tested.** In the 80 wrong answers the name is never right (city right in 12); most invent a typical pet name and city ("Bella", "Buddy", "Fluffy"; New York, London), some say the question is incomplete or loop ("cousin's cousin's ..."). Not tested here: the rest of the needle turn (owner, animal) is still evicted, and the model may need it to link the question to the kept name and city.

### 4. Archives and commits
- `/workspace/results_v6_npz.tar.gz` (15 MB, 137 .npz: n = 48 (120), n = 6 partial (1), logs_check_v6 with its JSONLs (16)) downloaded to development/results_v6/, md5 285dbe857f721f49cb290124603f4c31 (matches the server). Not in git.
- JSONLs and logs md5-checked against the server copies; committed and pushed from the local clone. The instance was left running (not destroyed).

### Amendment 1 (whole needle turn), in progress
- Commit 0931f5b brought to the server as a git bundle (390474d..main). The server's untracked v6 files (identical to b7ab572) were moved to /workspace/v6_untracked_before_amend1 to allow the fast-forward; the checked-out copies were byte-identical (cmp).
- `scripts/exp_v6.sh breadth_turn` (logs/exp_v6_breadth_turn.log, snapshot at 08:02) started 07:11:25.
- **B2' (manipulation check): MET.** `kvp_ft_turn_heads48` (whole needle user turn force-kept in all 48 heads), conditioned on full:0 correct: **117/119 = 0.983** (k = 4: 59/59, k = 8: 58/60; unconditional 60/60, 57/60). n = 48 took 07:11:25-07:46:12.
- **Binding (descriptive, paired):** `kvp_ft_turn_heads48` vs `kvp_ft_heads48`, same 119 dialogues, pooled k = 4, 8: only turn right 78, only value right 0, exact McNemar p = 6.6e-24 (k = 4: 25 vs 0; k = 8: 53 vs 0). results/paired_stats_v6_binding.md (run on the local PC from the md5-checked JSONLs). The tokens that bind the value to the question, not the value tokens alone, are what must survive.
- `kvp_ft_turn_heads6` finished 07:46:12-08:22:12 (120 dialogues): conditioned on full:0 correct **5/119** (k = 4: 5/59, k = 8: 0/60); n = 0 (v5 kvp_ft) 2/119; n = 48 117/119. Downloaded with its .npz (development/results_v6/turn6_0820/v6_turn6_npz.tar.gz, md5 1eb921a7bd8e9ea3216a06aa8955e115) and md5-checked.
- n = 12 started 08:22:12; **partial**: results/logs_eval_v6_breadth/kvp_ft_turn_heads12_b256_bf16.jsonl holds the 64 of 120 dialogues done at the 08:37:59 snapshot (evict_eval.py resumes from it; the 64 .npz files are in development/results_v6/turn12_snap4/v6_turn12_snap4_npz.tar.gz, md5 f61a00116ed3c6d69aab1935c7914c58). n = 24 not started. The vast.ai credit ran out, so **B1' (needs n = 6, 12, 24, 48) is not computed yet**.
- All downloads md5-checked against the server: turn n = 48 .npz (results_v6/partial_0802/v6_turn48_npz.tar.gz, ae32a9ab9b4b6098588443c48dced686), turn n = 6 .npz (above), value-only results_v6_npz.tar.gz (above). Not in git.
