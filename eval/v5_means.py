"""Per-config means of needle-survival and word-scoring fields (batch v5, rules 3 and 6). CPU only.

Rows per (method, budget, k) plus a pooled row over --pool_ks. Fields missing from a config's records
(e.g. name_ok on digit sets) are shown as '-'.

usage: python v5_means.py --logs results/logs_eval_v5_words_q1 --condition_on full:0 --out results/means_v5_words.md
"""
import argparse
from collections import defaultdict

from paired_stats import load

FIELDS = ["correct", "needle_at_answer", "needle_heads_full_at_answer", "name_ok", "city_ok", "city_hedged",
          "name_heads_full_at_answer", "city_heads_full_at_answer"]


def mean(recs, f):
    vals = [r[f] for r in recs if r.get(f) is not None]
    return f"{sum(vals) / len(vals):.3f}" if vals else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--pool_ks", default="4,8")
    ap.add_argument("--condition_on", default="",
                    help="METHOD:BUDGET, e.g. full:0 - keep only dialogues that config answered correctly")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    configs = load(args.logs)
    lines = []
    if args.condition_on:
        m, b = args.condition_on.rsplit(":", 1)
        ref = configs.get((m, int(b)))
        if ref is None:
            raise ValueError(f"--condition_on {args.condition_on}: no such config in {sorted(configs)}")
        keep = {i for i, r in ref.items() if r["correct"]}
        for key, recs in configs.items():
            if not set(recs) <= set(ref):
                raise ValueError(f"config {key} has dialogues missing from {args.condition_on}")
            configs[key] = {i: r for i, r in recs.items() if i in keep}
        lines += [f"Conditioned on {args.condition_on} correct: {len(keep)} / {len(ref)} dialogues kept.", ""]

    pool = {int(k) for k in args.pool_ks.split(",") if k}
    lines += ["| method | b | k | n | " + " | ".join(FIELDS) + " |", "|---" * (len(FIELDS) + 4) + "|"]
    for (m, b), recs in sorted(configs.items()):
        by_k = defaultdict(list)
        for r in recs.values():
            by_k[r["k"]].append(r)
        groups = [(str(k), by_k[k]) for k in sorted(by_k)]
        if pool:
            groups.append(("+".join(map(str, sorted(pool))), [r for k in sorted(pool) for r in by_k.get(k, [])]))
        for label, rs in groups:
            if rs:
                lines.append(f"| {m} | {b} | {label} | {len(rs)} | " + " | ".join(mean(rs, f) for f in FIELDS) + " |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        open(args.out, "w").write(text)


if __name__ == "__main__":
    main()
