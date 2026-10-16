"""Accuracy with 95% Wilson CIs and exact paired McNemar tests between eval configs.

Every config runs the same MT-NIAH dialogues, so method comparisons are paired by dialogue id.
p-values are Holm-adjusted across all tests in one run.

usage: python paired_stats.py --logs results/logs_eval results/logs_eval_v2 \
           --pairs kvp:h2o kvp:streaming kvp_pos600:kvp --out results/paired_stats.md
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson(c, n, z=1.96):
    p = c / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, mid - half), min(1.0, mid + half)


def mcnemar_exact(n10, n01):
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def load(dirs):
    configs = {}
    for d in dirs:
        for jl in sorted(Path(d).glob("*.jsonl")):
            recs = [json.loads(l) for l in jl.open()]
            if not recs:
                raise ValueError(f"{jl} is empty")
            key = (recs[0]["method"], recs[0]["budget"])
            by_id = {r["id"]: r for r in recs}
            if len(by_id) != len(recs):
                raise ValueError(f"{jl} has duplicate dialogue ids")
            if key in configs:
                if set(by_id) & set(configs[key]):
                    raise ValueError(f"config {key} appears twice with overlapping dialogue ids ({jl})")
                configs[key].update(by_id)
            else:
                configs[key] = by_id
    return configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--pairs", nargs="+", default=["kvp:h2o", "kvp:streaming"])
    ap.add_argument("--out", default="")
    ap.add_argument("--pool_ks", default="",
                    help="also one pre-registered test per pair and budget pooling these k values, e.g. 4,8 "
                         "(reported separately, not Holm-adjusted)")
    ap.add_argument("--condition_on", default="",
                    help="METHOD:BUDGET, e.g. full:0 - keep only dialogues that config answered correctly")
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

    lines += ["## Accuracy (95% Wilson CI)", "", "| method | b | k | correct / n | acc | CI |", "|---|---|---|---|---|---|"]
    for (m, b), recs in sorted(configs.items()):
        by_k = defaultdict(list)
        for r in recs.values():
            by_k[r["k"]].append(r["correct"])
        for k, cs in sorted(by_k.items()):
            lo, hi = wilson(sum(cs), len(cs))
            lines.append(f"| {m} | {b} | {k} | {sum(cs)} / {len(cs)} | {sum(cs) / len(cs):.3f} | [{lo:.2f}, {hi:.2f}] |")

    tests = []
    for pair in args.pairs:
        a, c = pair.split(":")
        budgets = sorted({b for (m, b) in configs if m == a} & {b for (m, b) in configs if m == c})
        if not budgets:
            raise ValueError(f"pair {pair}: no budget where both configs exist")
        for b in budgets:
            ra, rc = configs[(a, b)], configs[(c, b)]
            if set(ra) != set(rc):
                raise ValueError(f"pair {pair} b={b}: dialogue id sets differ")
            for k in sorted({r["k"] for r in ra.values()}):
                ids = [i for i, r in ra.items() if r["k"] == k]
                n10 = sum(ra[i]["correct"] and not rc[i]["correct"] for i in ids)
                n01 = sum(rc[i]["correct"] and not ra[i]["correct"] for i in ids)
                tests.append({"a": a, "c": c, "b": b, "k": k, "n": len(ids), "n10": n10, "n01": n01,
                              "p": mcnemar_exact(n10, n01)})
    for t, padj in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = padj

    lines += ["", f"## Paired exact McNemar ({len(tests)} tests, Holm-adjusted)", "",
              "| A vs B | b | k | n | only A right | only B right | p | p (Holm) |", "|---|---|---|---|---|---|---|---|"]
    for t in tests:
        lines.append(f"| {t['a']} vs {t['c']} | {t['b']} | {t['k']} | {t['n']} | {t['n10']} | {t['n01']} | "
                     f"{t['p']:.2g} | {t['p_holm']:.2g} |")
    if args.pool_ks:
        pool = {int(k) for k in args.pool_ks.split(",")}
        lines += ["", f"## Pre-registered pooled test (k in {sorted(pool)}, not adjusted)", "",
                  "| A vs B | b | n | only A right | only B right | p |", "|---|---|---|---|---|---|"]
        for pair in args.pairs:
            a, c = pair.split(":")
            for b in sorted({b for (m, b) in configs if m == a} & {b for (m, b) in configs if m == c}):
                ra, rc = configs[(a, b)], configs[(c, b)]
                ids = [i for i, r in ra.items() if r["k"] in pool]
                if not ids:
                    raise ValueError(f"pair {pair} b={b}: no dialogues with k in {sorted(pool)}")
                n10 = sum(ra[i]["correct"] and not rc[i]["correct"] for i in ids)
                n01 = sum(rc[i]["correct"] and not ra[i]["correct"] for i in ids)
                lines.append(f"| {a} vs {c} | {b} | {len(ids)} | {n10} | {n01} | {mcnemar_exact(n10, n01):.2g} |")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
