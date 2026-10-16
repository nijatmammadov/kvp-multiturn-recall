"""Batch v5 exploratory analyses (not pre-registered; descriptive only). CPU only.

A. Pooling at b=256 on the digit sets (seed-0, lead-2): per k, paired counts kvp_ft_pool{4,8} vs kvp_ft
   (only-pooled-right / only-kvp_ft-right) and means of needle_at_answer / needle_heads_full_at_answer per config.
B. Words vs digits at matched length: kvp and kvp_ft at b=256, accuracy and needle_at_answer binned by total_len
   (bins of 150 tokens); words conditioned on the full cache answering correctly, digits = lead-2.
   Plus, for words, the share of answers with name_ok only, city_ok only, both, neither.
C. Echoes: the full-cache reply to the needle turn, decoded from the full run's .npz token stream, flagged for
   name_echo / city_echo (words) or code_echo (digits, lead-2); accuracy of kvp, h2o, streaming, kvp_ft at b=256
   split by echo group. Needs the full_b0_bf16/ .npz directories and the tokenizer (CPU).

usage: python v5_exploratory.py --seed0 results/logs_eval_v4 --lead2 results/logs_eval_v4_lead2 \
           --words results/logs_eval_v5_words_q1 --tokenizer $P/models/Qwen2.5-0.5B-Instruct \
           --out results/v5_exploratory.md
A missing config raises an error; --partial (test runs only) prints it as missing instead.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from paired_stats import load

BIN = 150
WORD_CONFIGS = [("full", 0), ("kvp", 256), ("h2o", 256), ("streaming", 256)] + \
    [(m, b) for b in (256, 128) for m in ("kvp_ft", "kvp_ft_pool4", "kvp_ft_pool8")]
ECHO_METHODS = ("kvp", "h2o", "streaming", "kvp_ft")
ASSISTANT, IM_END = "<|im_start|>assistant\n", "<|im_end|>"


def get(configs, key, where, partial):
    if key in configs:
        return configs[key]
    if partial:
        return None
    raise ValueError(f"{where}: config {key[0]} b={key[1]} is missing")


def mean(rs, f):
    return sum(r[f] for r in rs) / len(rs)


def pooling(name, configs, partial):
    base = get(configs, ("kvp_ft", 256), name, partial)
    if base is None:
        return [f"### {name}: kvp_ft b=256 missing", ""]
    lines = [f"### {name}", "",
             "| config | k | n | acc | only pooled right | only kvp_ft right | needle_at_answer | needle_heads_full_at_answer |",
             "|---|---|---|---|---|---|---|---|"]
    for m in ("kvp_ft", "kvp_ft_pool4", "kvp_ft_pool8"):
        recs = get(configs, (m, 256), name, partial)
        if recs is None:
            lines.append(f"| {m} | missing | | | | | | |")
            continue
        if set(recs) != set(base):
            raise ValueError(f"{name}: {m} and kvp_ft cover different dialogues")
        by_k = defaultdict(list)
        for i, r in recs.items():
            by_k[r["k"]].append((r, base[i]))
        for k in sorted(by_k):
            pairs = by_k[k]
            rs = [r for r, _ in pairs]
            a = sum(r["correct"] and not b["correct"] for r, b in pairs)
            b = sum(b["correct"] and not r["correct"] for r, b in pairs)
            disc = "-" if m == "kvp_ft" else a
            disc_b = "-" if m == "kvp_ft" else b
            lines.append(f"| {m} | {k} | {len(rs)} | {mean(rs, 'correct'):.3f} | {disc} | {disc_b} | "
                         f"{mean(rs, 'needle_at_answer'):.3f} | {mean(rs, 'needle_heads_full_at_answer'):.3f} |")
    return lines + [""]


def binned(words, digits, method, partial):
    def bins(recs):
        out = defaultdict(list)
        for r in recs.values():
            out[r["total_len"] // BIN * BIN].append(r)
        return out

    w, d = get(words, (method, 256), "words", partial), get(digits, (method, 256), "lead-2", partial)
    if w is None or d is None:
        return [f"### {method}, b=256: missing", ""]
    w, d = bins(w), bins(d)
    lines = [f"### {method}, b=256", "",
             "| total_len | words n | words acc | words needle_at_answer | digits n | digits acc | digits needle_at_answer |",
             "|---|---|---|---|---|---|---|"]
    for lo in sorted(set(w) | set(d)):
        cells = []
        for g in (w.get(lo, []), d.get(lo, [])):
            cells += [str(len(g)), f"{mean(g, 'correct'):.3f}", f"{mean(g, 'needle_at_answer'):.3f}"] if g else ["0", "-", "-"]
        lines.append(f"| {lo}-{lo + BIN - 1} | " + " | ".join(cells) + " |")
    return lines + [""]


def word_parts(words):
    lines = ["### Words: name vs city (conditioned on full:0 correct)", "",
             "| method | b | k | n | name only | city only | both | neither |", "|---|---|---|---|---|---|---|---|"]
    for (m, b), recs in sorted(words.items()):
        if m == "full":
            continue
        by_k = defaultdict(list)
        for r in recs.values():
            by_k[r["k"]].append(r)
        for k in sorted(by_k):
            rs = by_k[k]
            share = lambda f: f"{sum(map(f, rs)) / len(rs):.3f}"
            lines.append(f"| {m} | {b} | {k} | {len(rs)} | "
                         f"{share(lambda r: r['name_ok'] and not r['city_ok'])} | "
                         f"{share(lambda r: r['city_ok'] and not r['name_ok'])} | "
                         f"{share(lambda r: r['name_ok'] and r['city_ok'])} | "
                         f"{share(lambda r: not r['name_ok'] and not r['city_ok'])} |")
    return lines + [""]


def needle_reply(tok, npz, rec, parts):
    """Full-cache reply to the needle turn (turn index = rec['lead']), checked against the stream's turn layout."""
    text = tok.decode(np.load(npz)["stream"].tolist())
    segs = text.split(ASSISTANT)
    n_turns = len(np.load(npz)["events"]) // 2 + 1
    if len(segs) != n_turns + 1:
        raise RuntimeError(f"{npz}: {len(segs) - 1} assistant headers, expected {n_turns}")
    turn = rec["lead"]
    user = segs[turn].rsplit("<|im_start|>user\n", 1)
    if len(user) != 2 or not all(p in user[1] for p in parts):
        raise RuntimeError(f"{npz}: user turn {turn} does not contain the needle {parts}")
    if IM_END not in segs[turn + 1]:
        raise RuntimeError(f"{npz}: reply to turn {turn} has no {IM_END}")
    return segs[turn + 1].split(IM_END, 1)[0]


def echoes(name, configs, tok, words, partial):
    full = get(configs, ("full", 0), name, False)
    npz_dir = Path(full[next(iter(full))]["_dir"]) / "full_b0_bf16"
    if not npz_dir.is_dir():
        raise RuntimeError(f"{name}: no full-cache .npz directory {npz_dir}")
    group = {}
    for i, r in full.items():
        npz = npz_dir / f"{i}.npz"
        if not npz.is_file():
            raise RuntimeError(f"{name}: {npz} missing")
        if words:
            nm, city = r["value"].split(" | ")
            reply = needle_reply(tok, npz, r, [nm, city]).lower()
            group[i] = ["none", "one", "name+city"][(nm.lower() in reply) + (city.lower() in reply)]
        else:
            reply = needle_reply(tok, npz, r, [r["value"]])
            group[i] = "code" if r["value"] in reply else "none"
    order = ["name+city", "one", "none"] if words else ["code", "none"]
    counts = {g: sum(v == g for v in group.values()) for g in order}
    lines = [f"### {name}", "", "Full-cache replies to the needle turn: " +
             ", ".join(f"{g} echoed {c}" for g, c in counts.items()) + f" (of {len(group)}).", "",
             "| method | b | k | " + " | ".join(f"{g}: correct / n" for g in order) + " |",
             "|---|---|---|" + "---|" * len(order)]
    for m in ECHO_METHODS:
        recs = get(configs, (m, 256), name, partial)
        if recs is None:
            lines.append(f"| {m} | 256 | missing |" + " |" * len(order))
            continue
        by_k = defaultdict(list)
        for i, r in recs.items():
            by_k[r["k"]].append(r)
        for k in sorted(by_k) + ["all"]:
            rs = list(recs.values()) if k == "all" else by_k[k]
            cells = []
            for g in order:
                sel = [r for r in rs if group[r["id"]] == g]
                cells.append(f"{sum(r['correct'] for r in sel)} / {len(sel)}")
            lines.append(f"| {m} | 256 | {k} | " + " | ".join(cells) + " |")
    return lines + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed0", required=True)
    ap.add_argument("--lead2", required=True)
    ap.add_argument("--words", required=True)
    ap.add_argument("--tokenizer", required=True, help="Qwen2.5-0.5B-Instruct directory (analysis C)")
    ap.add_argument("--out", default="")
    ap.add_argument("--partial", action="store_true", help="test runs only: show missing configs instead of failing")
    args = ap.parse_args()
    seed0, lead2, words = load([args.seed0]), load([args.lead2]), load([args.words])
    for configs, d in ((lead2, args.lead2), (words, args.words)):
        for recs in configs.values():
            for r in recs.values():
                r["_dir"] = d
    for key in WORD_CONFIGS:
        get(words, key, "words", args.partial)
    ref = get(words, ("full", 0), "words", False)
    keep = {i for i, r in ref.items() if r["correct"]}
    words = {key: {i: r for i, r in recs.items() if i in keep} for key, recs in words.items()}

    lines = ["# Batch v5: exploratory analyses (not pre-registered)", "",
             "Descriptive only; not part of the rule verdicts.", "",
             "## A. Pooling at b=256, digit sets", "",
             "Paired against kvp_ft on the same dialogues. needle_at_answer = share of needle tokens kept (mean over "
             "heads); needle_heads_full_at_answer = share of heads holding every needle token.", ""]
    lines += pooling("seed-0 (results/logs_eval_v4)", seed0, args.partial)
    lines += pooling("lead-2 (results/logs_eval_v4_lead2)", lead2, args.partial)
    lines += ["## B. Words vs digits at matched length", "",
              f"Words: {len(keep)} / {len(ref)} dialogues the full cache answered correctly. Digits: lead-2 set, all "
              f"dialogues. Bins of {BIN} tokens of total_len.", ""]
    lines += binned(words, lead2, "kvp", args.partial)
    lines += binned(words, lead2, "kvp_ft", args.partial)
    lines += word_parts(words)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    lines += ["## C. Echoes of the needle in the full-cache reply to the needle turn", "",
              "Each dialogue is grouped by whether the full-cache model repeated the needle when it answered the "
              "needle turn (case-insensitive substring of the decoded reply; digits: the 5-digit value). Accuracy of "
              "each method at b=256 per group. Words conditioned on full:0 correct; digits = lead-2 (full cache 180/180).", ""]
    lines += echoes("words (q1, conditioned on full:0 correct)", words, tok, True, args.partial)
    lines += echoes("digits lead-2", lead2, tok, False, args.partial)
    text = "\n".join(lines)
    print(text)
    if args.out:
        open(args.out, "w").write(text)


if __name__ == "__main__":
    main()
