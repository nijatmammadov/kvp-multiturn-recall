"""Batch v6, breadth test (rules B1, B2): accuracy vs number of heads in which the needle is force-kept.

n = 0 is the v5 kvp_ft_b256 run; n > 0 are the kvp_ft_heads{n}_b256 runs. Words set, conditioned on the full cache
answering correctly, k in --ks only. Cochran-Armitage trend test (one-sided, increasing) on the score n.

usage: python eval/head_breadth.py --base results/logs_eval_v5_words_q1 --breadth results/logs_eval_v6_breadth \
           --ns 6 12 24 48 --out results/head_breadth_v6.md
"""
import argparse
import json
import math
from pathlib import Path

from paired_stats import mcnemar_exact, wilson


def load(path):
    recs = {}
    for line in Path(path).open():
        r = json.loads(line)
        if r["id"] in recs:
            raise ValueError(f"{path}: duplicate id {r['id']}")
        recs[r["id"]] = r
    return recs


def trend_test(levels):
    """levels: [(score, correct, n)]. One-sided Cochran-Armitage test for an increasing trend. Returns (z, p)."""
    N = sum(n for _, _, n in levels)
    p = sum(c for _, c, _ in levels) / N
    if p in (0.0, 1.0):
        return 0.0, 1.0
    t = sum(s * (c - n * p) for s, c, n in levels)
    var = p * (1 - p) * (sum(n * s * s for s, _, n in levels) - sum(n * s for s, _, n in levels) ** 2 / N)
    z = t / math.sqrt(var)
    return z, 0.5 * math.erfc(z / math.sqrt(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="dir with full_b0_bf16.jsonl and kvp_ft_b256_bf16.jsonl")
    ap.add_argument("--breadth", required=True, help="dir with kvp_ft_heads{n}_b256_bf16.jsonl")
    ap.add_argument("--ns", nargs="+", type=int, required=True)
    ap.add_argument("--ks", default="4,8")
    ap.add_argument("--prefix", default="kvp_ft_heads", help="config prefix, e.g. kvp_ft_turn_heads (amendment 1)")
    ap.add_argument("--dtype", default="bf16", help="file suffix of every config, e.g. fp32 for the Kaggle ladder")
    ap.add_argument("--protected", default="name + city tokens", help="what is force-kept, for the report")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ks = {int(k) for k in args.ks.split(",")}

    full = load(Path(args.base) / f"full_b0_{args.dtype}.jsonl")
    ids = sorted(i for i, r in full.items() if r["correct"] and r["k"] in ks)
    runs = {0: load(Path(args.base) / f"kvp_ft_b256_{args.dtype}.jsonl")}
    for n in args.ns:
        runs[n] = load(Path(args.breadth) / f"{args.prefix}{n}_b256_{args.dtype}.jsonl")
        if runs[n][next(iter(runs[n]))]["protect_heads"] != n:
            raise ValueError(f"heads{n}: records say protect_heads={runs[n][next(iter(runs[n]))]['protect_heads']}")
    for n, recs in runs.items():
        missing = [i for i in ids if i not in recs]
        if missing:
            raise ValueError(f"n={n}: {len(missing)} conditioned dialogues missing, e.g. {missing[:3]}")
    for n in args.ns:
        low = [i for i in ids if runs[n][i]["needle_heads_full_at_answer"] < n / 48 - 1e-9]
        if low:
            raise RuntimeError(f"n={n}: needle held whole in fewer than {n} heads in {len(low)} dialogues, e.g. {low[:3]}")

    lines = ["# Head-breadth test (batch v6, rules B1-B2)", "",
             f"Words set (q1), b = 256, kvp_ft agents, k in {sorted(ks)}, conditioned on full-cache correct: "
             f"{len(ids)} dialogues per level. Force-kept ({args.protected}) in n of 48 (layer, KV head) pairs, "
             f"configs {args.prefix}{{n}}; "
             f"n = 0 is {Path(args.base) / f'kvp_ft_b256_{args.dtype}.jsonl'} (no protection).", "",
             "| n heads | correct / n | acc | 95% CI | name_ok | city_ok | heads whole (mean) | needle survival (mean) |",
             "|---|---|---|---|---|---|---|---|"]
    levels = []
    for n in sorted(runs):
        rs = [runs[n][i] for i in ids]
        c = sum(r["correct"] for r in rs)
        lo, hi = wilson(c, len(rs))
        mean = lambda f: sum(r[f] for r in rs) / len(rs)
        lines.append(f"| {n} | {c} / {len(rs)} | {c / len(rs):.3f} | [{lo:.2f}, {hi:.2f}] | {mean('name_ok'):.3f} | "
                     f"{mean('city_ok'):.3f} | {mean('needle_heads_full_at_answer'):.3f} | {mean('needle_at_answer'):.3f} |")
        levels.append((n, c, len(rs)))
    z, p = trend_test(levels)
    top = max(args.ns)
    acc_top = levels[-1][1] / levels[-1][2]
    a = sum(runs[top][i]["correct"] and not runs[0][i]["correct"] for i in ids)
    b = sum(runs[0][i]["correct"] and not runs[top][i]["correct"] for i in ids)
    lines += ["", f"**B2 (manipulation check, read first): accuracy at n = {top} = {acc_top:.3f} (needs >= 0.80) -> "
              f"{'MET' if acc_top >= 0.8 else 'NOT MET: force-keeping the needle tokens does not restore recall, so B1 cannot be read as a breadth effect'}.**",
              f"**B1 (primary): Cochran-Armitage trend over n = {sorted(runs)}: z = {z:.2f}, one-sided p = {p:.2g} -> "
              f"{'MET' if p < 0.05 else 'NOT MET'}.**",
              "", f"Descriptive: n = {top} vs n = 0, exact McNemar: only n={top} right {a}, only n=0 right {b}, "
              f"p = {mcnemar_exact(a, b):.2g}."]
    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
