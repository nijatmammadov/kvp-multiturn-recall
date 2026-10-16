"""Batch v7: offload + retrieve (rules R0, F1, F2; secondary S1, S2; descriptive).

Per set (words q1, digits lead-2), fp32, b = 256, k in {4, 8}, conditioned on the set's fp32 full-cache run:
kvp_ft (plain), kvp_ft_offload (evicted K/V kept on CPU; best-matching earlier user turn restored into all heads at
each new user turn, constant cache size), h2o, and the oracle kvp_ft_turn_heads48.

usage: python eval/offload_v7.py --words results/logs_eval_v6k_fp32 --digits results/logs_eval_v6k_fp32_lead2 \
           --out results/offload_v7.md
"""
import argparse
import json
from pathlib import Path

from paired_stats import holm, mcnemar_exact, wilson

CFG = {"kvp_ft": "kvp_ft_b256", "offload": "kvp_ft_offload_b256", "h2o": "h2o_b256", "oracle": "kvp_ft_turn_heads48_b256"}


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
        r = runs["offload"][i]
        if not r.get("offload_retrieve") or "retrieval_hit" not in r:
            raise RuntimeError(f"{d} offload {i}: record lacks offload fields")
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

    lines = ["# Offload + retrieve (batch v7)", "",
             "b = 256, k in {4, 8}, fp32 (Kaggle T4), conditioned on each set's full-cache run answering correctly.", "",
             "| set | config | correct / n | acc | 95% CI |", "|---|---|---|---|---|"]
    for s, (ids, runs) in sets.items():
        for k in CFG:
            c = sum(runs[k][i]["correct"] for i in ids)
            lo, hi = wilson(c, len(ids))
            lines.append(f"| {s} | {k} | {c} / {len(ids)} | {c / len(ids):.3f} | [{lo:.2f}, {hi:.2f}] |")

    lines += [""]
    for s, (ids, runs) in sets.items():
        o = runs["offload"]
        hits = [i for i in ids if o[i]["retrieval_hit"]]
        nr = [o[i]["n_restored"] for i in ids]
        lines.append(f"R0 ({s}): retrieval picked the needle turn at the question in {len(hits)} / {len(ids)}; "
                     f"accuracy when hit {sum(o[i]['correct'] for i in hits)} / {len(hits)}, when missed "
                     f"{sum(o[i]['correct'] for i in ids if i not in hits)} / {len(ids) - len(hits)}; "
                     f"restored tokens at the question: mean {sum(nr) / len(nr):.1f}.")

    prim = [pair(sets[s][1], sets[s][0], "offload", "kvp_ft") for s in ("words", "digits")]
    ph = holm([t[2] for t in prim])
    sec = [pair(sets[s][1], sets[s][0], "offload", "h2o") for s in ("words", "digits")]
    sh = holm([t[2] for t in sec])
    met = lambda t, p: "MET" if p < 0.05 and t[0] > t[1] else "NOT MET"
    lines += [""]
    for name, s, t, p in (("F1", "words", prim[0], ph[0]), ("F2", "digits", prim[1], ph[1])):
        lines.append(f"**{name} (primary, {s}): offload vs kvp_ft: only offload right {t[0]}, only kvp_ft right {t[1]}, "
                     f"p = {t[2]:.2g}, p_holm = {p:.2g} -> {met(t, p)}.**")
    for name, s, t, p in (("S1", "words", sec[0], sh[0]), ("S2", "digits", sec[1], sh[1])):
        lines.append(f"{name} (secondary, {s}): offload vs h2o: only offload right {t[0]}, only h2o right {t[1]}, "
                     f"p = {t[2]:.2g}, p_holm = {p:.2g} -> {met(t, p)} (offload better).")
    for s, (ids, runs) in sets.items():
        t = pair(runs, ids, "oracle", "offload")
        lines.append(f"Descriptive ({s}): oracle (turn kept in all heads) vs offload: only oracle right {t[0]}, "
                     f"only offload right {t[1]}, p = {t[2]:.2g}.")
    lines += ["", "Caveat (pre-registered): retrieval is lexical and the benchmark questions repeat the needle's words, "
              "so R0 is near-perfect by construction; this measures the eviction side, not retrieval under paraphrase."]
    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
