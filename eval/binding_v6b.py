"""Batch v6 amendment 3: which part of the needle turn must survive (rules D0, K1, D1; descriptive K2, D2).

All configs: kvp_ft agents, b = 256, protection in all 48 heads, k in {4, 8}, fp32, conditioned on the same set's
full-cache run answering correctly. value = name + city (words) or code (digits); keyval = value + linking phrase
row["key"]; turn = the whole needle user turn.

usage: python eval/binding_v6b.py --words results/logs_eval_v6k_fp32 --digits results/logs_eval_v6k_fp32_lead2 \
           --out results/binding_v6b.md
"""
import argparse
import json
from pathlib import Path

from paired_stats import holm, mcnemar_exact, wilson

CFG = {"value": "kvp_ft_heads48_b256", "keyval": "kvp_ft_keyval_heads48_b256", "turn": "kvp_ft_turn_heads48_b256"}


def load(path):
    recs = {}
    for line in Path(path).open():
        r = json.loads(line)
        if r["id"] in recs:
            raise ValueError(f"{path}: duplicate id {r['id']}")
        recs[r["id"]] = r
    return recs


def load_set(d, dtype):
    full = load(Path(d) / f"full_b0_{dtype}.jsonl")
    runs = {k: load(Path(d) / f"{v}_{dtype}.jsonl") for k, v in CFG.items()}
    ids = sorted(i for i, r in full.items() if r["correct"] and r["k"] in (4, 8))
    for k, recs in runs.items():
        missing = [i for i in ids if i not in recs]
        if missing:
            raise ValueError(f"{d} {k}: {len(missing)} conditioned dialogues missing, e.g. {missing[:3]}")
    for i in ids:
        kv = runs["keyval"][i]
        if not kv["n_forced"] > kv["n_needle_tokens"]:
            raise RuntimeError(f"{d} keyval {i}: n_forced {kv['n_forced']} <= value tokens {kv['n_needle_tokens']}")
        if kv["needle_heads_full_at_answer"] != 1.0:
            raise RuntimeError(f"{d} keyval {i}: value not whole in all heads")
    return ids, runs


def pair(runs, ids, a, b):
    n10 = sum(runs[a][i]["correct"] and not runs[b][i]["correct"] for i in ids)
    n01 = sum(runs[b][i]["correct"] and not runs[a][i]["correct"] for i in ids)
    return n10, n01, mcnemar_exact(n10, n01)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", required=True)
    ap.add_argument("--digits", required=True)
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sets = {"words": load_set(args.words, args.dtype), "digits": load_set(args.digits, args.dtype)}

    lines = ["# Which part of the needle turn must survive (batch v6, amendment 3)", "",
             "kvp_ft, b = 256, protection in all 48 heads, k in {4, 8}, fp32 (Kaggle T4), conditioned on full-cache correct.",
             "", "| set | protected | correct / n | acc | 95% CI | mean forced tokens |", "|---|---|---|---|---|---|"]
    for s, (ids, runs) in sets.items():
        for k in ("value", "keyval", "turn"):
            c = sum(runs[k][i]["correct"] for i in ids)
            lo, hi = wilson(c, len(ids))
            nf = [runs[k][i].get("n_forced") for i in ids]
            nf = f"{sum(nf) / len(nf):.1f}" if all(x is not None for x in nf) else "n/a (older run)"
            lines.append(f"| {s} | {k} | {c} / {len(ids)} | {c / len(ids):.3f} | [{lo:.2f}, {hi:.2f}] | {nf} |")

    ids_d, runs_d = sets["digits"]
    d0 = sum(runs_d["turn"][i]["correct"] for i in ids_d) / len(ids_d)
    ids_w, runs_w = sets["words"]
    k1 = pair(runs_w, ids_w, "keyval", "value")
    d1 = pair(runs_d, ids_d, "turn", "value")
    ph = holm([k1[2], d1[2]])
    verdict = lambda t, p: "MET" if p < 0.05 and t[0] > t[1] else "NOT MET"
    k2 = pair(runs_w, ids_w, "keyval", "turn")
    acc = lambda runs, ids, k: sum(runs[k][i]["correct"] for i in ids) / len(ids)
    lines += ["",
              f"**D0 (manipulation check, read first): digits turn accuracy = {d0:.3f} (needs >= 0.80) -> "
              f"{'MET' if d0 >= 0.8 else 'NOT MET: D1 cannot be read'}.**",
              f"**K1 (primary, words): keyval vs value: only keyval right {k1[0]}, only value right {k1[1]}, "
              f"p = {k1[2]:.2g}, p_holm = {ph[0]:.2g} -> {verdict(k1, ph[0])}.**",
              f"**D1 (primary, digits): turn vs value: only turn right {d1[0]}, only value right {d1[1]}, "
              f"p = {d1[2]:.2g}, p_holm = {ph[1]:.2g} -> {verdict(d1, ph[1])}.**", "",
              f"K2 (descriptive, words): keyval {acc(runs_w, ids_w, 'keyval'):.3f} vs turn {acc(runs_w, ids_w, 'turn'):.3f} "
              f"(ratio {acc(runs_w, ids_w, 'keyval') / max(acc(runs_w, ids_w, 'turn'), 1e-9):.2f}); paired: only keyval right "
              f"{k2[0]}, only turn right {k2[1]}, p = {k2[2]:.2g}. Ratio >= 0.90: the linking phrase is enough; "
              "lower: other words of the turn are needed too."]
    for a, b in (("keyval", "value"), ("keyval", "turn")):
        t = pair(runs_d, ids_d, a, b)
        lines.append(f"D2 (descriptive, digits): {a} vs {b}: only {a} right {t[0]}, only {b} right {t[1]}, p = {t[2]:.2g}.")
    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
