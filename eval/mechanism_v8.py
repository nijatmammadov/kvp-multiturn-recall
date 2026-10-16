"""Batch v8: is the mechanism a small-model effect? Same tests on Qwen2.5-1.5B-Instruct and 0.5B, StreamingLLM base.

Per model (words q1, b = 256, k in {4, 8}, fp32, conditioned on that model's full-cache run answering correctly):
streaming (nothing protected), streaming + value (name + city in all heads), + keyval (value + linking phrase),
+ turn (whole needle turn), + offload (best-matching earlier user turn restored into all heads; lexical retriever).
Rules M0-M3 on 1.5B (primary, Holm over M1-M3); the same tests on 0.5B are descriptive; cross-size value-only.

usage: python eval/mechanism_v8.py --big results/logs_eval_v8_1p5b_fp32 --small results/logs_eval_v8_0p5b_fp32 \
           --out results/mechanism_v8.md
"""
import argparse
import json
from pathlib import Path

from paired_stats import holm, mcnemar_exact, wilson

CFG = {"streaming": "streaming_b256", "value": "streaming_value_b256", "keyval": "streaming_keyval_b256",
       "turn": "streaming_turn_b256", "offload": "streaming_offload_b256"}


def load(path):
    recs = {}
    for line in Path(path).open():
        r = json.loads(line)
        if r["id"] in recs:
            raise ValueError(f"{path}: duplicate id {r['id']}")
        recs[r["id"]] = r
    return recs


def load_model_dir(d, dtype):
    full = load(Path(d) / f"full_b0_{dtype}.jsonl")
    runs = {k: load(Path(d) / f"{v}_{dtype}.jsonl") for k, v in CFG.items()}
    kept = sorted(i for i, r in full.items() if r["k"] in (4, 8))
    ids = [i for i in kept if full[i]["correct"]]
    models = {r.get("model") for recs in runs.values() for r in recs.values()} | {r.get("model") for r in full.values()}
    if len(models) != 1:
        raise RuntimeError(f"{d}: mixed models {models}")
    for k, recs in runs.items():
        missing = [i for i in kept if i not in recs]
        if missing:
            raise ValueError(f"{d} {k}: {len(missing)} dialogues missing, e.g. {missing[:3]}")
    for i in ids:
        for k in ("value", "keyval", "turn"):
            if runs[k][i]["needle_heads_full_at_answer"] != 1.0:
                raise RuntimeError(f"{d} {k} {i}: needle not whole in all heads")
    return models.pop(), full, kept, ids, runs


def pair(a, b, ids):
    n10 = sum(a[i]["correct"] and not b[i]["correct"] for i in ids)
    n01 = sum(b[i]["correct"] and not a[i]["correct"] for i in ids)
    return n10, n01, mcnemar_exact(n10, n01)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", required=True)
    ap.add_argument("--small", required=True)
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sets = {"big": load_model_dir(args.big, args.dtype), "small": load_model_dir(args.small, args.dtype)}

    lines = ["# Is the mechanism a small-model effect? (batch v8)", "",
             "Words q1, StreamingLLM base, b = 256 per head, k in {4, 8}, fp32 (Kaggle T4); protection in all heads. "
             "Each model conditioned on its own full-cache run.", "",
             "| model | full cache (all k = 4, 8) | config | correct / n | acc | 95% CI |", "|---|---|---|---|---|---|"]
    for _, (model, full, kept, ids, runs) in sets.items():
        fc = f"{sum(full[i]['correct'] for i in kept)} / {len(kept)}"
        for k in CFG:
            c = sum(runs[k][i]["correct"] for i in ids)
            lo, hi = wilson(c, len(ids))
            lines.append(f"| {model} | {fc} | {k} | {c} / {len(ids)} | {c / len(ids):.3f} | [{lo:.2f}, {hi:.2f}] |")

    model, full, kept, ids, runs = sets["big"]
    m0 = sum(full[i]["correct"] for i in kept) / len(kept)
    tests = [("M1", "turn vs value", pair(runs["turn"], runs["value"], ids)),
             ("M2", "keyval vs value", pair(runs["keyval"], runs["value"], ids)),
             ("M3", "offload vs streaming", pair(runs["offload"], runs["streaming"], ids))]
    ph = holm([t[2][2] for t in tests])
    lines += ["", f"**M0 (read first, {model}): full cache {m0:.3f} on k = 4, 8 (needs >= 0.80) -> "
                  f"{'MET' if m0 >= 0.8 else 'NOT MET: the word set is not usable for this model; M1-M3 not read'}.**"]
    for (name, label, t), p in zip(tests, ph):
        lines.append(f"**{name} ({model}): {label}: only first right {t[0]}, only second right {t[1]}, p = {t[2]:.2g}, "
                     f"p_holm = {p:.2g} -> {'MET' if p < 0.05 and t[0] > t[1] else 'NOT MET'}.**")
    hits = sum(runs["offload"][i].get("retrieval_hit", False) for i in ids)
    lines.append(f"Offload retrieval hit at the question ({model}): {hits} / {len(ids)}.")

    s_model, _, _, s_ids, s_runs = sets["small"]
    lines += ["", f"Descriptive, same tests on {s_model} (not adjusted):"]
    for label, a, b in (("turn vs value", "turn", "value"), ("keyval vs value", "keyval", "value"),
                        ("offload vs streaming", "offload", "streaming")):
        t = pair(s_runs[a], s_runs[b], s_ids)
        lines.append(f"- {label}: only first right {t[0]}, only second right {t[1]}, p = {t[2]:.2g}")
    both = sorted(set(ids) & set(s_ids))
    for k in ("value", "keyval", "streaming"):
        t = pair(runs[k], s_runs[k], both)
        lines.append(f"Cross-size ({k}, {len(both)} dialogues both models' full cache got right): only {model} right "
                     f"{t[0]}, only {s_model} right {t[1]}, p = {t[2]:.2g}.")
    lines += ["", "Reading: the mechanism holds at the larger size if M1-M3 are met there. If value alone reaches the "
              "turn accuracy at 1.5B, the need for the linking context was a small-model effect."]
    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
