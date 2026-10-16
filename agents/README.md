# Trained KVP agents (Qwen2.5-0.5B-Instruct, 24 layers x 2 KV heads)

- `oasst2_0_5b/`: the original 48 agents (v1), trained on OASST2 multi-turn traces, 4,000 steps each.
- `oasst2_recall_ft_0_5b/`: the 48 warm-start fine-tuned agents (batch v4), 1,500 steps on OASST2 replay + recall traces.

Each `layer_XXXXXX/kv_head_XXX/best_ckpt.pth` is the best-eval checkpoint (model + EMA weights, config, metrics history). The evals use the EMA weights.

Use them with:

```bash
python eval/chat_evict.py --methods kvp,full --budget 128 --agents_dir agents/oasst2_recall_ft_0_5b
python eval/evict_eval.py --method kvp --budget 256 --agents_dir agents/oasst2_0_5b
```

Not included (kept locally): the periodic training checkpoints of the fine-tune (resume-only, 724 MB) and the trace data (regenerate with the configs in `repo_changes/`).
