"""Recompute needle survival as fraction of needle-value tokens still in each head's cache.

evict_eval.survival() computed isin(cache_rows, needle).mean() = fraction of cache rows that
are needle tokens. Here: isin(needle, kept_rows).mean() per (layer, head), from the saved
`kept` arrays (padded with -1). Read-only: writes nothing but stdout.
"""
import json
from collections import defaultdict

import numpy as np
from transformers import AutoTokenizer

from evict_eval import find_needle, turn_tokens

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
rows = {json.loads(l)["id"]: json.loads(l) for l in open("mt_niah.jsonl")}
needles = {i: np.array(find_needle(tok, turn_tokens(tok, r, 0), r["value"])) for i, r in rows.items()}

print("| method | b | k=2 | k=4 | k=8 |")
print("|---|---|---|---|---|")
for m, b in [("streaming", 128), ("streaming", 256), ("h2o", 128), ("h2o", 256)]:
    name = f"{m}_b{b}_bf16"
    recs = [json.loads(l) for l in open(f"logs/{name}.jsonl")]
    by_k = defaultdict(list)
    for r in recs:
        z = np.load(f"logs/{name}/{r['id']}.npz")
        if str(z["events"][-1]) != f"t{len(rows[r['id']]['user_turns']) - 1}_prefill":
            raise RuntimeError(f"{r['id']}: last event {z['events'][-1]}")
        kept = z["kept"][-1]  # [24, 2, b], at answer time
        surv = np.array([[np.isin(needles[r["id"]], kept[l, h]).mean() for h in range(kept.shape[1])]
                         for l in range(kept.shape[0])])
        by_k[r["k"]].append((r["correct"], surv.mean(), (surv == 1.0).mean(), (surv > 0).mean()))
    cells = []
    for k in (2, 4, 8):
        a = np.array(by_k[k], dtype=float)
        cells.append(f"{a[:, 0].mean():.3f} / {a[:, 1].mean():.3f} / {a[:, 2].mean():.3f}")
    print(f"| {m} | {b} | " + " | ".join(cells) + " |")
print("\ncells = accuracy / mean fraction of needle-value tokens kept (avg over 48 heads) / fraction of heads keeping all 5")
