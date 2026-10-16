"""MT-NIAH eviction analysis from saved .npz logs.

Recomputes needle survival from kept positions (the JSONL survival fields from the first
heuristic grid are wrong), counts echoed copies of the code in the model's own replies,
and reports what each method keeps at answer time.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CATS = ["template", "system", "needle_turn", "filler_user", "question", "assistant"]
BUCKETS = [(0, 0), (1, 4), (5, 16), (17, 48)]
COLORS = {"full": "#2a78d6", "streaming": "#eb6834", "h2o": "#1baf7a", "kvp": "#eda100"}
MARKERS = {"full": "o", "streaming": "s", "h2o": "^", "kvp": "D"}
LABELS = {"full": "Full cache", "streaming": "StreamingLLM", "h2o": "H2O", "kvp": "KVP"}


def wilson(c, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = c / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, mid - half), min(1.0, mid + half)


class Decoder:
    def __init__(self, tok):
        self.tok = tok
        self.cache = {}
        self.im_start = tok.convert_tokens_to_ids("<|im_start|>")
        self.im_end = tok.convert_tokens_to_ids("<|im_end|>")

    def piece(self, t):
        if t not in self.cache:
            self.cache[t] = self.tok.decode([int(t)])
        return self.cache[t]


def segment(stream, dec):
    """Category per position, plus the start position of the final user turn."""
    cats = [None] * len(stream)
    role, user_turn, after_role = None, -1, False
    user_starts = []
    for i, t in enumerate(stream):
        t = int(t)
        if t == dec.im_start:
            cats[i] = "template"
            role = dec.piece(stream[i + 1]).strip()
            if role not in ("system", "user", "assistant"):
                raise ValueError(f"unexpected role token {role!r} at {i + 1}")
            if role == "user":
                user_turn += 1
                user_starts.append(i)
            continue
        if i > 0 and int(stream[i - 1]) == dec.im_start:
            cats[i] = "template"
            after_role = True
            continue
        if after_role and dec.piece(t) == "\n":
            cats[i] = "template"
            after_role = False
            continue
        after_role = False
        if t == dec.im_end or (i > 0 and int(stream[i - 1]) == dec.im_end and dec.piece(t) == "\n"):
            cats[i] = "template"
            continue
        if role == "system":
            cats[i] = "system"
        elif role == "assistant":
            cats[i] = "assistant"
        elif role == "user":
            cats[i] = "needle_turn" if user_turn == 0 else "user"
        else:
            raise ValueError(f"token at {i} before any role")
    last_user = user_starts[-1]
    cats = ["question" if c == "user" and i >= last_user else ("filler_user" if c == "user" else c)
            for i, c in enumerate(cats)]
    return cats, last_user, user_starts


def occurrences(stream, dec, value, limit):
    pieces = [dec.piece(t) for t in stream[:limit]]
    n = len(value)
    out = []
    for i in range(len(pieces) - n + 1):
        w = pieces[i:i + n]
        if all(p.strip().isdigit() for p in w) and "".join(w).replace(" ", "") == value:
            out.append(list(range(i, i + n)))
    return out


def analyze_dialogue(npz, rec, row, dec):
    stream = npz["stream"]
    cats, last_user, user_starts = segment(stream, dec)
    occ = occurrences(stream, dec, row["value"], last_user)
    if not occ or not (user_starts[0] <= occ[0][0] < (user_starts[1] if len(user_starts) > 1 else last_user)):
        raise RuntimeError(f"{row['id']}: needle not found in the first user turn")
    events = [str(e) for e in npz["events"]]
    ans = events.index(f"t{row['k'] + 1}_prefill")
    kept = npz["kept"][ans]
    n_l, n_h, _ = kept.shape
    mask = np.zeros((n_l, n_h, len(stream)), dtype=bool)
    for l in range(n_l):
        for h in range(n_h):
            k = kept[l, h]
            mask[l, h, k[k >= 0]] = True
    needle = occ[0]
    needle_frac = mask[:, :, needle].mean(-1)
    full_copy = np.stack([mask[:, :, o].all(-1) for o in occ]).any(0)
    cat_arr = np.array(cats)
    n_seen = int(kept.max()) + 1
    base = {c: float((cat_arr[:n_seen] == c).mean()) for c in CATS}
    comp = {}
    for c in CATS:
        sel = cat_arr == c
        comp[c] = float((mask[:, :, sel].sum(-1) / mask.sum(-1)).mean())
    return {
        "correct": bool(rec["correct"]), "echo": len(occ) > 1, "needle_frac": needle_frac,
        "needle_full": needle_frac == 1.0, "any_full": full_copy, "comp": comp,
        "base": base,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="/workspace/kv/logs")
    ap.add_argument("--data", default="/workspace/kv/mt_niah.jsonl")
    ap.add_argument("--out", default="/workspace/kv/analysis")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(args.data)}
    dec = Decoder(AutoTokenizer.from_pretrained(MODEL))
    logs = Path(args.logs)

    summary, acc_rows = [], []
    head_stats = defaultdict(lambda: [0, 0, 0, 0])
    for jl in sorted(logs.glob("*.jsonl")):
        recs = [json.loads(l) for l in jl.open()]
        method, budget = recs[0]["method"], recs[0]["budget"]
        for k in sorted({r["k"] for r in recs}):
            rk = [r for r in recs if r["k"] == k]
            c = sum(r["correct"] for r in rk)
            lo, hi = wilson(c, len(rk))
            acc_rows.append({"method": method, "budget": budget, "k": k, "n": len(rk), "acc": c / len(rk),
                             "ci_lo": lo, "ci_hi": hi})
        if method == "full":
            continue
        per_k = defaultdict(list)
        head_sum = defaultdict(lambda: 0.0)
        for r in recs:
            npz = np.load(jl.parent / jl.stem / f"{r['id']}.npz")
            if "stream" not in npz.files:
                raise RuntimeError(f"{jl.stem}/{r['id']}: no stream saved; re-run this config")
            d = analyze_dialogue(npz, r, rows[r["id"]], dec)
            per_k[r["k"]].append(d)
            for (l, h), full in np.ndenumerate(d["any_full"]):
                s = head_stats[(l, h)]
                s[0 if full else 2] += 1
                s[1 if full else 3] += d["correct"]
            if r["k"] == 8:
                for (l, h), f in np.ndenumerate(d["needle_frac"]):
                    head_sum[(l, h)] += f
        n8 = len(per_k[8])
        if n8:
            with open(out / f"head_needle_survival_{jl.stem}_k8.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["layer", "head", "mean_needle_survival"])
                for (l, h), v in sorted(head_sum.items()):
                    w.writerow([l, h, round(v / n8, 4)])
        for k, ds in sorted(per_k.items()):
            echo = [d for d in ds if d["echo"]]
            noecho = [d for d in ds if not d["echo"]]
            row = {
                "method": method, "budget": budget, "k": k, "n": len(ds),
                "acc": np.mean([d["correct"] for d in ds]),
                "n_echo": len(echo),
                "acc_echo": np.mean([d["correct"] for d in echo]) if echo else float("nan"),
                "acc_noecho": np.mean([d["correct"] for d in noecho]) if noecho else float("nan"),
                "needle_survival": np.mean([d["needle_frac"].mean() for d in ds]),
                "heads_needle_complete": np.mean([d["needle_full"].mean() for d in ds]),
                "heads_any_copy_complete": np.mean([d["any_full"].mean() for d in ds]),
            }
            for lo, hi in BUCKETS:
                sel = [d for d in ds if lo <= d["any_full"].sum() <= hi]
                row[f"acc_heads_{lo}-{hi}"] = f"{np.mean([d['correct'] for d in sel]):.2f} (n={len(sel)})" if sel else "-"
            for c in CATS:
                row[f"kept_{c}"] = np.mean([d["comp"][c] for d in ds])
            for c in CATS:
                row[f"base_{c}"] = np.mean([d["base"][c] for d in ds])
            summary.append(row)

    with open(out / "accuracy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(acc_rows[0]))
        w.writeheader()
        w.writerows(acc_rows)
    if summary:
        with open(out / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)

    imp = []
    for (l, h), (n_f, c_f, n_n, c_n) in sorted(head_stats.items()):
        if n_f >= 10 and n_n >= 10:
            imp.append({"layer": l, "head": h, "n_complete": n_f, "acc_complete": c_f / n_f,
                        "n_missing": n_n, "acc_missing": c_n / n_n, "diff": c_f / n_f - c_n / n_n})
    imp.sort(key=lambda r: -r["diff"])
    if imp:
        with open(out / "head_importance.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(imp[0]))
            w.writeheader()
            w.writerows(imp)

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) and not math.isnan(v) else str(v)

    lines = ["# MT-NIAH analysis", "", "## Accuracy (95% Wilson CI)", "",
             "| method | b | k | n | acc | CI |", "|---|---|---|---|---|---|"]
    for r in acc_rows:
        lines.append(f"| {r['method']} | {r['budget']} | {r['k']} | {r['n']} | {r['acc']:.3f} | "
                     f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] |")
    if summary:
        keys = list(summary[0])
        lines += ["", "## Survival, echoes, kept composition (at answer time)", "",
                  "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
        for r in summary:
            lines.append("| " + " | ".join(fmt(r[k]) for k in keys) + " |")
    if imp:
        lines += ["", "## Heads whose complete copy of the code best predicts a correct answer "
                  "(pooled over eviction configs; correlational)", "",
                  "| layer | head | acc if complete (n) | acc if missing (n) | diff |", "|---|---|---|---|---|"]
        for r in imp[:10]:
            lines.append(f"| {r['layer']} | {r['head']} | {r['acc_complete']:.2f} ({r['n_complete']}) | "
                         f"{r['acc_missing']:.2f} ({r['n_missing']}) | {r['diff']:+.2f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot(acc_rows, out)


def plot(acc_rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budgets = sorted({r["budget"] for r in acc_rows if r["method"] != "full"})
    ks = sorted({r["k"] for r in acc_rows})
    fig, axes = plt.subplots(1, len(budgets), figsize=(4.2 * len(budgets), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, b in zip(axes, budgets):
        for m in ["full", "streaming", "h2o", "kvp"]:
            rs = sorted([r for r in acc_rows if r["method"] == m and (m == "full" or r["budget"] == b)],
                        key=lambda r: r["k"])
            if not rs:
                continue
            x = [r["k"] for r in rs]
            y = [r["acc"] for r in rs]
            err = [[max(0.0, r["acc"] - r["ci_lo"]) for r in rs], [max(0.0, r["ci_hi"] - r["acc"]) for r in rs]]
            ax.errorbar(x, y, yerr=err, color=COLORS[m], marker=MARKERS[m], markersize=8, linewidth=2,
                        capsize=3, elinewidth=1, label=LABELS[m])
            ax.annotate(LABELS[m], (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color="#52514e")
        ax.set_title(f"budget b = {b} tokens per head", fontsize=10, color="#0b0b0b")
        ax.set_xticks(ks)
        ax.set_xlabel("filler turns k", color="#52514e")
        ax.set_xlim(min(ks) - 0.5, max(ks) + 2.5)
        ax.set_ylim(-0.03, 1.05)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_facecolor("#fcfcfb")
    axes[0].set_ylabel("exact-match accuracy", color="#52514e")
    axes[-1].legend(frameon=False, fontsize=8, loc="center right")
    fig.patch.set_facecolor("#fcfcfb")
    fig.suptitle("MT-NIAH retrieval vs filler turns (Qwen2.5-0.5B-Instruct, 60 dialogues per point, 95% CI)",
                 fontsize=10, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(out / "accuracy_vs_k.png", dpi=200)
    print(f"\nfigure: {out / 'accuracy_vs_k.png'}")


if __name__ == "__main__":
    main()
