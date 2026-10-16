"""Re-test KVP vs baselines after dropping dialogues whose filler prompts occur in the training data.

usage: python overlap_robustness.py --out results/overlap_robustness.md
"""
import argparse
import json
from pathlib import Path

from paired_stats import mcnemar_exact

CELLS = [(256, 2), (256, 4), (128, 2), (128, 4)]
BASELINES = [("h2o", "logs_eval"), ("streaming", "logs_eval"), ("h2o_std", "logs_eval_v2")]


def load(path):
    return {json.loads(l)["id"]: json.loads(l) for l in open(path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).resolve().parent.parent / "results"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    res = Path(args.results)

    bad = set(json.loads((res / "overlap_check.json").read_text())["overlapping_fillers"])
    rows = [json.loads(l) for l in open(res / "mt_niah.jsonl")]
    excluded = {r["id"] for r in rows if any(t in bad for t in r["user_turns"][1:-1])}

    lines = [f"## KVP vs baselines without the {len(excluded)} dialogues whose filler occurs in training", "",
             "| b | k | baseline | subset | n | KVP acc | baseline acc | only KVP right | only baseline right | p (exact McNemar) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for b, k in CELLS:
        kvp = load(res / "logs_eval" / f"kvp_b{b}_bf16.jsonl")
        for base, folder in BASELINES:
            path = res / folder / f"{base}_b{b}_bf16.jsonl"
            if not path.exists():
                continue
            other = load(path)
            if set(other) != set(kvp):
                raise ValueError(f"{path}: dialogue ids differ from kvp_b{b}")
            ids_all = [i for i, r in kvp.items() if r["k"] == k]
            for subset, ids in (("all", ids_all), ("no overlap", [i for i in ids_all if i not in excluded])):
                n10 = sum(kvp[i]["correct"] and not other[i]["correct"] for i in ids)
                n01 = sum(other[i]["correct"] and not kvp[i]["correct"] for i in ids)
                acc_a = sum(kvp[i]["correct"] for i in ids) / len(ids)
                acc_b = sum(other[i]["correct"] for i in ids) / len(ids)
                lines.append(f"| {b} | {k} | {base} | {subset} | {len(ids)} | {acc_a:.3f} | {acc_b:.3f} | "
                             f"{n10} | {n01} | {mcnemar_exact(n10, n01):.2g} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
