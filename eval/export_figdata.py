"""Export per-token head counts for the report figures from evict_eval .npz logs.

For every config directory <name>_fp32/ (with its <name>_fp32.jsonl) under --logs and every dialogue in it: which of
the (layer, KV head) caches hold each token of the fact turn and of the question when the question arrives (the last
logged event, t{n-1}_prefill), plus the model's answer. The full cache has no kept arrays: every head holds every token.

usage: python eval/export_figdata.py --logs results/logs_figs_fp32 --data results/mt_niah_words_lead2_q1.jsonl \
           --out results/figdata.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def user_turn_spans(pieces, ids, im_start, im_end):
    """Token index ranges [a, b) of the content of each user turn in the stream."""
    spans = []
    i = 0
    while i < len(ids):
        if ids[i] != im_start:
            i += 1
            continue
        j, head = i + 1, ""
        while j < len(ids) and not head.endswith("\n"):
            head += pieces[j]
            j += 1
        if head.strip() == "user":
            k = j
            while k < len(ids) and ids[k] != im_end:
                k += 1
            if k == len(ids):
                raise RuntimeError(f"user turn starting at token {i} has no <|im_end|>")
            spans.append((j, k))
            i = k
        else:
            i = j
    return spans


def char_roles(pieces, texts_roles):
    """Role per token ('answer', 'link' or '') from the first occurrence of each text in the joined pieces."""
    starts, n = [], 0
    for p in pieces:
        starts.append(n)
        n += len(p)
    full = "".join(pieces)
    roles = [""] * len(pieces)
    for text, role in texts_roles:
        at = full.find(text)
        if at == -1:
            raise RuntimeError(f"{text!r} not found in {full!r}")
        end = at + len(text)
        for t, s in enumerate(starts):
            if s < end and s + len(pieces[t]) > at:
                roles[t] = role
    return roles


def head_sets(kept_last):
    """kept_last: [layers, heads, budget] positions (-1 = empty) -> list of sets, rows ordered L0.H0, L0.H1, ..."""
    n_l, n_h, _ = kept_last.shape
    return [set(int(p) for p in kept_last[l, h] if p >= 0) for l in range(n_l) for h in range(n_h)]


def export_dialogue(npz_path, row, rec, tok, im_start, im_end, n_heads_full):
    z = np.load(npz_path)
    ids = [int(t) for t in z["stream"]]
    pieces = [tok.decode([t]) for t in ids]
    n_turns = len(row["user_turns"])
    events = [str(e) for e in z["events"]]
    if events[-1] != f"t{n_turns - 1}_prefill":
        raise RuntimeError(f"{npz_path}: last event {events[-1]!r}, expected t{n_turns - 1}_prefill")

    spans = user_turn_spans(pieces, ids, im_start, im_end)
    if len(spans) != n_turns:
        raise RuntimeError(f"{npz_path}: found {len(spans)} user turns, dialogue has {n_turns}")
    for i, (a, b) in enumerate(spans):
        got = tok.decode(ids[a:b]).strip()
        if got != row["user_turns"][i].strip():
            raise RuntimeError(f"{npz_path}: user turn {i} decodes to {got[:60]!r}, expected {row['user_turns'][i][:60]!r}")

    if "kept" in z.files:
        sets = head_sets(z["kept"][-1])
    else:
        sets = None

    def segment(a, b, roles=None):
        positions = list(range(a, b))
        if sets is None:
            counts = [n_heads_full] * len(positions)
            matrix = None
        else:
            counts = [sum(p in s for s in sets) for p in positions]
            matrix = [[int(p in s) for p in positions] for s in sets]
        out = {"pieces": pieces[a:b], "counts": counts, "matrix": matrix}
        if roles is not None:
            out["roles"] = roles
        return out

    fa, fb = spans[row["needle_turn"]]
    roles = char_roles(pieces[fa:fb], [(row["key"], "link"), (row["name"], "answer"), (row["city"], "answer")])
    qa, qb = spans[-1]
    return {
        "answer": rec["answer"], "correct": bool(rec["correct"]),
        "fact": segment(fa, fb, roles), "question": segment(qa, qb),
        "n_heads": n_heads_full if sets is None else len(sets),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    from transformers import AutoConfig, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    n_heads_full = cfg.num_hidden_layers * cfg.num_key_value_heads
    im_start, im_end = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(args.data)}

    logs = Path(args.logs)
    configs = sorted(p.name[: -len("_fp32.jsonl")] for p in logs.glob("*_fp32.jsonl"))
    if not configs:
        raise SystemExit(f"no *_fp32.jsonl in {logs}")
    out = {"model": args.model, "n_heads": n_heads_full, "dialogues": {}}
    for name in configs:
        recs = {json.loads(l)["id"]: json.loads(l) for l in (logs / f"{name}_fp32.jsonl").open()}
        for did, rec in recs.items():
            npz = logs / f"{name}_fp32" / f"{did}.npz"
            if not npz.is_file():
                raise RuntimeError(f"missing {npz}")
            d = out["dialogues"].setdefault(did, {"k": rows[did]["k"], "name": rows[did]["name"],
                                                  "city": rows[did]["city"], "key": rows[did]["key"], "configs": {}})
            d["configs"][name] = export_dialogue(npz, rows[did], rec, tok, im_start, im_end, n_heads_full)
        print(f"{name}: {len(recs)} dialogues", flush=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"wrote {args.out}: {len(out['dialogues'])} dialogues x {len(configs)} configs")


if __name__ == "__main__":
    main()
