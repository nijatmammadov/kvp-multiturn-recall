import json

import numpy as np
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
rows = {json.loads(l)["id"]: json.loads(l) for l in open("mt_niah.jsonl")}
old = {json.loads(l)["id"]: json.loads(l) for l in open("logs_smoke/h2o_b128_bf16.jsonl")}
new = {json.loads(l)["id"]: json.loads(l) for l in open("logs_smoke_stream/h2o_b128_bf16.jsonl")}
for i in ("k2_000", "k2_001"):
    r = rows[i]
    s = np.load(f"logs_smoke_stream/h2o_b128_bf16/{i}.npz")["stream"].tolist()
    turns = tok.decode(s, skip_special_tokens=False).split("<|im_start|>assistant\n")
    replies = [t.split("<|im_end|>")[0] for t in turns[1:]]
    print(f"== {i} value {r['value']} | final answer: smoke {old[i]['answer']!r}, re-run {new[i]['answer']!r}")
    print(f"needle: {r['needle']}")
    print(f"turn-1 reply: {replies[0]!r}")
    print(f"code in turn-1 reply: {r['value'] in replies[0]}; in replies: {[n for n, x in enumerate(replies, 1) if r['value'] in x]}")
