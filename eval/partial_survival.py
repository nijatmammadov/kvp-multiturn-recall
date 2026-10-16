"""Needle fragmentation: how many of the L needle-value tokens each (layer, head) still holds at answer time.

Reads the per-dialogue .npz files written by evict_eval.py (v2 or later; their `needle` field is the
per-head share of value tokens kept after each eviction event). CPU only.

usage: python partial_survival.py --logs results/logs_eval_v2 results/logs_eval_v3 --out results/fragmentation.md
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

L_VALUE = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    hist = defaultdict(lambda: np.zeros(L_VALUE + 1, dtype=np.int64))
    acc = defaultdict(lambda: [0, 0])
    for d in args.logs:
        for jl in sorted(Path(d).glob("*.jsonl")):
            recs = [json.loads(l) for l in jl.open()]
            npz_dir = jl.with_suffix("")
            if recs[0]["method"] == "full":
                continue
            if not npz_dir.is_dir():
                print(f"skip {jl.name}: no .npz directory")
                continue
            for r in recs:
                z = np.load(npz_dir / f"{r['id']}.npz")
                events = [str(e) for e in z["events"]]
                k = int(re.search(r"k(\d+)_\d+$", r["id"]).group(1))
                if k != r["k"]:
                    raise ValueError(f"{r['id']}: id says k={k}, record says {r['k']}")
                ans = events.index(f"t{r.get('lead', 0) + k + 1}_prefill")
                counts = np.rint(z["needle"][ans] * L_VALUE).astype(int).ravel()
                if counts.min() < 0 or counts.max() > L_VALUE:
                    raise ValueError(f"{r['id']}: needle counts out of range")
                key = (r["method"] + (f"_lead{r['lead']}" if r.get("lead") else ""), r["budget"], k)
                hist[key] += np.bincount(counts, minlength=L_VALUE + 1)
                acc[key][0] += r["correct"]
                acc[key][1] += 1

    lines = ["## Needle fragmentation at answer time", "",
             f"Share of (dialogue, layer, head) pairs holding 0..{L_VALUE} of the {L_VALUE} value tokens. "
             "'Fragmented' = holding some but not all.", "",
             "| method | b | k | n dialogues | acc | " + " | ".join(f"{i}/{L_VALUE}" for i in range(L_VALUE + 1))
             + " | fragmented |", "|---|---|---|---|---|" + "---|" * (L_VALUE + 2)]
    for key in sorted(hist):
        h = hist[key] / hist[key].sum()
        c, n = acc[key]
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {n} | {c / n:.3f} | "
                     + " | ".join(f"{x:.3f}" for x in h) + f" | {h[1:L_VALUE].sum():.3f} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
