"""Recall prompts for training KVP agents to keep facts that are asked about later (batch v4).

Goes in src/kvcompression/data/. Reads the prompts built by eval/make_recall_prompts.py (data/recall_prompts.jsonl
in the kvp-eviction repo: short RULER-style needle-in-a-haystack documents over WikiText-2, held out from MT-NIAH)
and tokenizes them for unroll_and_store.py. The assistant turn is pre-filled with the answer prefix, so the
generated tokens are the value(s) and their attention lands on the needle.
"""
import json

import torch
from torchtune.data import Message


class RecallNIAHDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, prompts_file, min_tokens=300, max_tokens=2200, limit=0):
        super().__init__()
        self.tokenizer = tokenizer
        rows = [json.loads(l) for l in open(prompts_file, encoding="utf-8")]
        if [r["id"] for r in rows] != list(range(len(rows))):
            raise ValueError(f"{prompts_file}: ids must be 0..n-1 in order")
        if limit:
            rows = rows[:limit]
        self.samples, self.answers, self.variants = [], [], []
        for r in rows:
            messages = [Message(role="system", content=r["system"]),
                        Message(role="user", content=r["user"]),
                        Message(role="assistant", content=r["answer_prefix"], eot=False)]
            tokens, mask = tokenizer.tokenize_messages(messages, add_eos=False)
            if not min_tokens <= len(tokens) <= max_tokens:
                raise ValueError(f"prompt {r['id']}: {len(tokens)} tokens outside [{min_tokens}, {max_tokens}]")
            self.samples.append((tokens, mask))
            self.answers.append(r["answers"])
            self.variants.append(r["variant"])
        lens = [len(t) for t, _ in self.samples]
        print(f"RecallNIAHDataset: {len(lens)} prompts, tokens min/mean/max "
              f"{min(lens)}/{sum(lens) // len(lens)}/{max(lens)}", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, mask = self.samples[idx]
        return {"tokens": tokens, "mask": mask, "id": idx}
