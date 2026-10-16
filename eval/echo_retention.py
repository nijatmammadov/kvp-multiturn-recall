"""Exploratory (not pre-registered), CPU only. Reads the per-dialogue .npz eviction logs.

1. Echo retention. For each method, finds every copy of the needle in that method's OWN token stream: the original
   (needle user turn) and echoes (the fact repeated in an assistant reply before the question). At answer time it
   measures, per (layer, KV head), whether each copy is held whole. Answers: does H2O win on words by keeping the
   echoed copy rather than the original?
2. Head coverage. From the logged needle survival per head: share of heads holding ALL needle tokens, ANY needle
   token, and the mean, per config and per survival bin. Answers: at matched mean survival, does pooling concentrate
   the needle into fewer heads?

Self-check: the original-copy numbers are recomputed from the kept indices and must equal the logged
needle_at_answer, needle_heads_full_at_answer and name/city_heads_full_at_answer of every record.

usage: python eval/echo_retention.py --tokenizer tokenizer.json \
           --words results/logs_eval_v5_words_q1 --words_npz <dir with the config subdirs> \
           --words_configs kvp_b256 h2o_b256 streaming_b256 kvp_ft_b256 kvp_ft_pool4_b256 kvp_ft_pool8_b256 \
           --digits results/logs_eval_v4_lead2 --digits_npz <dir> \
           --digits_configs kvp_ft_b256 kvp_ft_pool4_b256 kvp_ft_pool8_b256 \
           --out results/echo_retention_v5.md
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

TOL = 1e-6
SURV_BINS = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.01)]


def load_jsonl(path):
    recs = {}
    for line in path.open():
        r = json.loads(line)
        if r["id"] in recs:
            raise ValueError(f"{path}: duplicate id {r['id']}")
        recs[r["id"]] = r
    return recs


class Stream:
    """Token stream of one dialogue; stream index == cache position."""

    def __init__(self, tok, ids, n_user_turns):
        self.ids = ids
        im_start = tok.token_to_id("<|im_start|>")
        if im_start is None:
            raise ValueError("tokenizer has no <|im_start|>")
        self.pieces = [tok.decode([int(t)], skip_special_tokens=False) for t in ids]
        self.starts = np.cumsum([0] + [len(p) for p in self.pieces])
        heads = [i for i, t in enumerate(ids) if t == im_start]
        bounds = heads + [len(ids)]
        self.segs = [(bounds[j], bounds[j + 1]) for j in range(len(heads))]
        expect = 1 + 2 * n_user_turns
        if len(self.segs) != expect:
            raise RuntimeError(f"{len(self.segs)} <|im_start|> segments, expected {expect}")
        for j, role in enumerate(["system"] + ["user", "assistant"] * n_user_turns):
            if not self.text(*self.segs[j]).startswith(f"<|im_start|>{role}"):
                raise RuntimeError(f"segment {j} is not {role}: {self.text(*self.segs[j])[:40]!r}")

    def text(self, a, b):
        return "".join(self.pieces[a:b])

    def find(self, seg, needle):
        """Token positions of every case-insensitive occurrence of `needle` inside segment `seg`."""
        a, b = self.segs[seg]
        hay, out, at = self.text(a, b).lower(), [], 0
        base = self.starts[a]
        while (at := hay.find(needle.lower(), at)) != -1:
            lo, hi = base + at, base + at + len(needle)
            out.append([i for i in range(a, b) if self.starts[i] < hi and self.starts[i + 1] > lo])
            at += len(needle)
        return out


def copies(stream, row_turns, needle_turn, part):
    """(original copy, [echo copies]) of `part`, searched in every segment before the final question."""
    user_seg = lambda u: 1 + 2 * u
    q_seg = user_seg(row_turns - 1)
    orig = stream.find(user_seg(needle_turn), part)
    if len(orig) != 1:
        raise RuntimeError(f"{part!r}: {len(orig)} occurrences in the needle turn, expected 1")
    echoes = []
    for seg in range(1, q_seg):
        if seg == user_seg(needle_turn):
            continue
        found = stream.find(seg, part)
        if found and seg % 2 == 1:
            raise RuntimeError(f"{part!r} appears in user segment {seg} other than the needle turn")
        echoes += found
    return orig[0], echoes


def per_pos(kept, pos):
    """[L, H, len(pos)] bool: position held by that head. kept: [L, H, budget] with -1 padding."""
    return (kept[..., :, None] == np.asarray(pos)[None, None, None, :]).any(axis=-2)


def held(kept, pos):
    """[L, H] bool: head holds every position in pos."""
    if not pos:
        raise ValueError("empty copy")
    return per_pos(kept, pos).all(axis=-1)


def held_any(kept, pos):
    return np.isin(kept, np.asarray(pos)).any(axis=-1)


def analyse(tok, rec, npz, parts):
    z = np.load(npz)
    ids = z["stream"].astype(int).tolist()
    n_turns = rec["k"] + rec["lead"] + 2
    events = [str(e) for e in z["events"]]
    if events[-1] != f"t{n_turns - 1}_prefill":
        raise RuntimeError(f"{npz}: last event {events[-1]}, expected t{n_turns - 1}_prefill")
    kept = z["kept"][-1]
    s = Stream(tok, ids, n_turns)
    out, orig_all = {}, []
    for name, text in parts.items():
        orig, echoes = copies(s, n_turns, rec["lead"], text)
        orig_all += orig
        o = held(kept, orig)
        e = np.zeros_like(o)
        for c in echoes:
            e |= held(kept, c)
        out[name] = {"n_echo": len(echoes), "orig_whole": o.mean(), "echo_whole": e.mean(),
                     "any_whole": (o | e).mean(), "orig_tok": held_any(kept, orig).mean(),
                     "echo_tok": held_any(kept, [p for c in echoes for p in c]).mean() if echoes else 0.0,
                     "orig_hit": bool(o.any()), "echo_hit": bool(e.any())}
        logged = rec.get(f"{name}_heads_full_at_answer")
        if logged is not None and abs(logged - o.mean()) > TOL:
            raise RuntimeError(f"{npz}: {name} whole-head share {o.mean():.6f} != logged {logged:.6f}")
    surv = per_pos(kept, sorted(set(orig_all))).mean(-1)
    if abs(surv.mean() - rec["needle_at_answer"]) > TOL:
        raise RuntimeError(f"{npz}: needle survival {surv.mean():.6f} != logged {rec['needle_at_answer']:.6f}")
    if abs(z["needle"][-1].mean() - rec["needle_at_answer"]) > TOL:
        raise RuntimeError(f"{npz}: npz needle field disagrees with the record")
    return out


def coverage(npz, rec):
    n = np.load(npz)["needle"][-1]
    if abs(n.mean() - rec["needle_at_answer"]) > TOL or abs((n == 1).mean() - rec["needle_heads_full_at_answer"]) > TOL:
        raise RuntimeError(f"{npz}: needle field disagrees with the record")
    return {"mean": n.mean(), "whole": (n == 1).mean(), "any": (n > 0).mean()}


def config_records(logs, npz_root, cfg, keep=None):
    recs = load_jsonl(Path(logs) / f"{cfg}_bf16.jsonl")
    d = Path(npz_root) / f"{cfg}_bf16"
    if not d.is_dir():
        raise FileNotFoundError(f"no .npz directory {d}")
    if keep is not None:
        recs = {i: r for i, r in recs.items() if i in keep}
    return recs, d


def f3(x):
    return f"{x:.3f}"


def echo_section(title, tok, logs, npz_root, configs, keep, parts_of, part_names):
    lines = [f"### {title}", "",
             "Copy held whole = all its tokens in that (layer, head) at answer time. Echo = the fact repeated in an "
             "assistant reply of the SAME run before the question.", ""]
    hdr = "| config | n | echoed (all parts) | acc, echoed | acc, not echoed | " + " | ".join(
        f"{p} orig whole | {p} echo whole | {p} orig tok | {p} echo tok" for p in part_names) + " |"
    lines += [hdr, "|" + "---|" * (5 + 4 * len(part_names))]
    detail = []
    for cfg in configs:
        recs, d = config_records(logs, npz_root, cfg, keep)
        res = {i: analyse(tok, r, d / f"{i}.npz", parts_of(r)) for i, r in recs.items()}
        ech = {i: all(res[i][p]["n_echo"] > 0 for p in part_names) for i in recs}
        acc = lambda sel: f"{sum(recs[i]['correct'] for i in sel)} / {len(sel)}"
        m = lambda p, f: f3(np.mean([res[i][p][f] for i in recs]))
        lines.append(f"| {cfg} | {len(recs)} | {sum(ech.values())} | {acc([i for i in recs if ech[i]])} | "
                     f"{acc([i for i in recs if not ech[i]])} | " + " | ".join(
                         f"{m(p, 'orig_whole')} | {m(p, 'echo_whole')} | {m(p, 'orig_tok')} | {m(p, 'echo_tok')}"
                         for p in part_names) + " |")
        p0 = part_names[0]
        ok = lambda r: r.get(f"{p0}_ok", r["correct"])
        cells = []
        for o in (True, False):
            for e in (True, False):
                sel = [i for i in recs if res[i][p0]["orig_hit"] == o and res[i][p0]["echo_hit"] == e]
                cells.append(f"{sum(ok(recs[i]) for i in sel)} / {len(sel)}")
        detail.append(f"| {cfg} | " + " | ".join(cells) + " |")
    lines += ["", f"{part_names[0]} recalled, by which copy of the {part_names[0]} is held whole in at least one head:", "",
              "| config | original + echo | original only | echo only | neither |", "|---|---|---|---|---|"] + detail
    return lines + [""]


def coverage_section(title, logs, npz_root, configs, keep):
    lines = [f"### {title}", "", "| config | n | mean survival | heads whole | heads any | any / mean |",
             "|---|---|---|---|---|---|"]
    binrows = []
    for cfg in configs:
        recs, d = config_records(logs, npz_root, cfg, keep)
        cov = {i: coverage(d / f"{i}.npz", r) for i, r in recs.items()}
        mean = np.mean([c["mean"] for c in cov.values()])
        anyc = np.mean([c["any"] for c in cov.values()])
        lines.append(f"| {cfg} | {len(recs)} | {f3(mean)} | {f3(np.mean([c['whole'] for c in cov.values()]))} | "
                     f"{f3(anyc)} | {anyc / mean:.2f} |")
        for lo, hi in SURV_BINS:
            sel = [i for i in recs if lo <= cov[i]["mean"] < hi]
            if sel:
                binrows.append(f"| {cfg} | [{lo}, {min(hi, 1.0)}) | {len(sel)} | "
                               f"{sum(recs[i]['correct'] for i in sel)} | "
                               f"{f3(np.mean([cov[i]['whole'] for i in sel]))} | "
                               f"{f3(np.mean([cov[i]['any'] for i in sel]))} |")
    lines += ["", "Per mean-survival bin:", "", "| config | survival bin | n | correct | heads whole | heads any |",
              "|---|---|---|---|---|---|"] + binrows
    return lines + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="path to Qwen2.5-0.5B-Instruct tokenizer.json")
    ap.add_argument("--words", required=True)
    ap.add_argument("--words_npz", default="")
    ap.add_argument("--full", default="", help="full-cache JSONL for conditioning (default: <words>/full_b0_bf16.jsonl)")
    ap.add_argument("--words_configs", nargs="+", required=True)
    ap.add_argument("--digits", default="")
    ap.add_argument("--digits_npz", default="")
    ap.add_argument("--digits_configs", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)

    full = load_jsonl(Path(args.full) if args.full else Path(args.words) / "full_b0_bf16.jsonl")
    keep = {i for i, r in full.items() if r["correct"]}
    wn = args.words_npz or args.words
    words_parts = lambda r: dict(zip(("name", "city"), r["value"].split(" | ")))
    lines = ["# Echo retention and head coverage (v5, exploratory, not pre-registered)", "",
             f"Words: conditioned on the full cache answering correctly ({len(keep)} / {len(full)}). "
             "All original-copy numbers were recomputed from the kept indices and match the logged fields.", ""]
    lines += echo_section("Echoes, words", tok, args.words, wn, args.words_configs, keep, words_parts, ["name", "city"])
    if args.digits_configs:
        dn = args.digits_npz or args.digits
        lines += echo_section("Echoes, digits", tok, args.digits, dn, args.digits_configs, None,
                              lambda r: {"code": r["value"]}, ["code"])
    lines += coverage_section("Head coverage, words", args.words, wn, args.words_configs, keep)
    if args.digits_configs:
        lines += coverage_section("Head coverage, digits", args.digits, args.digits_npz or args.digits,
                                  args.digits_configs, None)
    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
