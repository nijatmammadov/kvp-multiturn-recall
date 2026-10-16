import json
from collections import defaultdict

import numpy as np
from transformers import AutoTokenizer

CFGS = [("full", 0), ("streaming", 128), ("streaming", 256), ("h2o", 128), ("h2o", 256)]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
rows = {json.loads(l)["id"]: json.loads(l) for l in open("mt_niah.jsonl")}


def turn1_echo(path, value):
    s = np.load(path)["stream"].tolist()
    reply1 = tok.decode(s, skip_special_tokens=False).split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0]
    return value in reply1


print("| method | b | k=2 | k=4 | k=8 |")
print("|---|---|---|---|---|")
echo_by_id = {}
detail = []
for m, b in CFGS:
    name = f"{m}_b{b}_bf16"
    recs = [json.loads(l) for l in open(f"logs/{name}.jsonl")]
    ids = [r["id"] for r in recs]
    if len(ids) != len(set(ids)) or len(ids) != 180:
        raise RuntimeError(f"{name}: {len(ids)} records, {len(set(ids))} unique")
    by_k = defaultdict(list)
    for r in recs:
        by_k[r["k"]].append(r)
    cells = []
    for k in (2, 4, 8):
        rs = by_k[k]
        acc = sum(r["correct"] for r in rs) / len(rs)
        surv = sum(r["needle_at_answer"] for r in rs) / len(rs)
        cells.append(f"{acc:.3f} / {surv:.3f}")
        if m != "full":
            ech = [turn1_echo(f"logs/{name}/{r['id']}.npz", r["value"]) for r in rs]
            for r, e in zip(rs, ech):
                echo_by_id[r["id"]] = e
            c_e = sum(r["correct"] for r, e in zip(rs, ech) if e)
            n_e = sum(ech)
            c_ne = sum(r["correct"] for r, e in zip(rs, ech) if not e)
            detail.append(f"{m} b={b} k={k}: turn-1 echo {n_e}/60; correct with echo {c_e}/{n_e}, correct without echo {c_ne}/{60 - n_e}")
    print(f"| {m} | {b} | " + " | ".join(cells) + " |")
print("\ncells = accuracy / mean needle survival at answer (fraction of needle-value tokens kept, averaged over 24x2 heads)\n")
print("\n".join(detail))
