# Why learned KV-cache eviction forgets facts in multi-turn dialogue

This repository evaluates Apple's learned KV-cache eviction policy KVP ("Learning to Evict", [arXiv 2602.10238](https://arxiv.org/abs/2602.10238), code: [apple/ml-learning-to-evict](https://github.com/apple/ml-learning-to-evict) @ `d33c2dc`) in multi-turn conversations, explains why it fails to recall earlier facts, and tests a fix derived from that explanation.

Models: Qwen2.5-0.5B-Instruct (48 KVP agents, 24 layers × 2 KV heads, trained here on OASST2) and Qwen2.5-1.5B-Instruct. Budget: 256 cached tokens per KV head. Baselines: full cache, StreamingLLM, H2O.

## Summary

1. **KVP almost never recalls a fact from earlier in the conversation.** On a multi-turn needle test (a fact stated in turn 3, 4 or 8 unrelated turns, then a question), KVP answers 4% of word facts and 4% of number facts correctly, against 100% for the full cache.
2. **Keeping the fact's own tokens is not enough. The model needs the phrase that binds the fact to its subject.** Forcing the cache to keep the answer tokens ("Lady Waffle", "Bergen") in all heads gives 0% (StreamingLLM base) to 31% (KVP base). Keeping the answer *plus* the linking phrase ("cousin's hamster") gives 79–99%; keeping the whole sentence gives 99–100%.
3. **The binding has to be present in (nearly) all heads.** Keeping the whole fact sentence in 6, 12, 24 or 48 of the 48 heads gives 6%, 10%, 36% and 100%.
4. **This holds at both model sizes** (0.5B and 1.5B) and in two precisions (bf16, fp32).
5. **A fix that follows from this:** keep evicted tokens in CPU memory, and when a new user message arrives, restore the best-matching earlier user message *whole and into all heads*, at the same GPU cache size. Recall goes from 4% to 100% (KVP base, 0.5B) and from 0% to 99% (StreamingLLM base, 1.5B). Restoring evicted content from CPU memory is a known family of methods (e.g. InfLLM, Quest); the contribution here is the diagnosis of *what* has to be restored and *where*.
6. **Needle-token survival, a common diagnostic for eviction methods, does not predict recall**: tokens of the fact can survive in many heads while recall stays near zero.

All tests use decision rules fixed before running, paired exact McNemar tests (Holm-adjusted) on the same dialogues, and Wilson 95% intervals. Accuracies below are conditioned on dialogues the full cache answers correctly.

## Main results

**The fix vs KVP and H2O** (Qwen2.5-0.5B, fp32, k = 4, 8 filler turns; [`results/offload_v7.md`](results/offload_v7.md)):

| needle | KVP | H2O | KVP + restore | upper bound (needle turn force-kept) |
|---|---|---|---|---|
| words (pet name + city) | 4 / 112 | 59 / 112 | **112 / 112** | 112 / 112 |
| digits (5-digit code) | 5 / 120 | 15 / 120 | **120 / 120** | 120 / 120 |

KVP + restore vs KVP: 108 vs 0 discordant dialogues (words), 115 vs 0 (digits), p_holm < 1e-32. Same peak GPU memory; ~26% slower.

**What has to survive** (force-kept in all 48 heads on top of KVP, 0.5B, fp32; [`results/binding_v6b.md`](results/binding_v6b.md)):

| needle | answer tokens only | answer + linking phrase | whole sentence |
|---|---|---|---|
| words | 35 / 112 | 88 / 112 | 112 / 112 |
| digits | 111 / 120 | 119 / 120 | 120 / 120 |

The digit filler turns contain no numbers, so a lone 5-digit number in the cache is unambiguous; the word set has no such shortcut.

**How many heads** (whole fact sentence force-kept in n random heads, KVP, 0.5B, words; [`results/head_breadth_v6k_fp32.md`](results/head_breadth_v6k_fp32.md)): n = 0, 6, 12, 24, 48 gives 4, 7, 11, 40, 112 of 112 (trend test p = 4e-74).

**Model size** (StreamingLLM base, words, fp32; [`results/mechanism_v8.md`](results/mechanism_v8.md)):

| model | StreamingLLM | + answer tokens | + answer + linking phrase | + whole sentence | + restore |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | 0 / 117 | 0 / 117 | 116 / 117 | 116 / 117 | 116 / 117 |
| Qwen2.5-0.5B | 0 / 112 | 0 / 112 | 105 / 112 | 112 / 112 | 111 / 112 |

**Finding the right message under paraphrase** ([`results/paraphrase_v7.md`](results/paraphrase_v7.md)): a word-overlap retriever (used in the GPU runs above) fails when the question shares fewer than two content words with the fact (0 / 180 on 4 of 10 paraphrase templates). A sentence-embedding retriever (all-MiniLM-L6-v2) picks the right message in 1,796 of 1,800 paraphrased questions. This was measured on retrieval alone, not end to end.

## Limitations

- One model family (Qwen2.5), 0.5B and 1.5B; KVP agents exist for 0.5B only.
- Synthetic conversations with one fact each; real conversations with several facts, or questions that need facts from different turns, are untested.
- The restore step brings back one message. In live chats it can pick an earlier *question* about the fact instead of the fact itself (see the demo below), and a fact split over two messages is only partly restored.
- CPU memory grows with conversation length.

## Live chat demo

Chat with the model while its cache is evicted after every turn, with several methods side by side:

```bash
pip install torch transformers==4.55.4 numpy omegaconf sentence-transformers
python eval/chat_evict.py --methods kvp,kvp+fix,full --budget 256 --agents_dir agents/oasst2_recall_ft_0_5b
```

`kvp+fix` is KVP with the restore step (`--retriever embed` by default, or `lexical`). Without KVP agents, e.g. on 1.5B: `--model Qwen/Qwen2.5-1.5B-Instruct --methods streaming,streaming+fix,full`. In the chat: `/find TEXT` shows how many heads still hold a phrase, `/memory kvp` shows what the cache holds, `/reset`, `/quit`.

Example (0.5B, fp32): after *"Please remember this for later: the access code for the silent harbor C6 is 32162"* and three unrelated questions, KVP answers 32164 (the five digits were together in 5 of 48 heads), KVP + restore and the full cache answer 32162.

## Layout

- `eval/evict_eval.py`: the eviction harness (full, streaming, h2o, h2o_std, tova, kvp; `--protect_needle value|keyvalue|turn`, `--protect_heads N`, `--offload_retrieve`, `--model`).
- `eval/offload.py`: word-overlap retriever. `eval/chat_evict.py`: live chat. `eval/mt_niah.py`: multi-turn needle sets.
- Analysis: `paired_stats.py`, `head_breadth.py`, `binding_v6b.py`, `offload_v7.py`, `mechanism_v8.py`, `paraphrase_check.py`, `echo_retention.py`, `partial_survival.py` and others in `eval/`.
- Tests (CPU, tiny random model, no downloads): `eval/test_*.py`.
- `scripts/`: run scripts for a single GPU server (`exp_v*.sh`) and for Kaggle (`kaggle_v*.py`, fp32).
- `agents/`: the trained KVP agents for Qwen2.5-0.5B (`oasst2_0_5b`: trained on OASST2; `oasst2_recall_ft_0_5b`: fine-tuned on OASST2 + recall traces, used in the main results).
- `repo_changes/`: files added to Apple's repository for training (datasets, configs, warm-start training).
- `results/`: datasets, per-dialogue JSONL results, analysis reports; [`results/RESULTS.md`](results/RESULTS.md) is the detailed log of batches v1–v6 (bf16). Per-dialogue `.npz` eviction logs and run logs are not included (large).
- `training/`, `training_ft/`: training curves and rewards of the agents.

## Reproducing

Main results (fp32, one T4 or better): clone apple/ml-learning-to-evict at `d33c2dc` next to this repository, `pip install transformers==4.55.4 omegaconf`, then e.g. `python scripts/kaggle_v7.py --apple ../ml-learning-to-evict --no_push` (the fix) or `python scripts/kaggle_v8.py --no_push` (model size; no agents needed). The harness resumes by dialogue id, so with the committed results in place these scripts only redo the analysis; to rerun the models, first delete the `results/logs_eval_*` directories they write to (`v6k_fp32`, `v6k_fp32_lead2`, `v8_*`). Single runs: `python eval/evict_eval.py --help`.

## Acknowledgements

KVP and its training code are by Apple ([apple/ml-learning-to-evict](https://github.com/apple/ml-learning-to-evict)), used and modified under Apple's license; files in `repo_changes/` are modified or added files for that repository, and the agents in `agents/` were trained with it. Filler turns come from OASST2 ([OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2)).
