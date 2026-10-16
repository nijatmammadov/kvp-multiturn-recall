import json
import re
import statistics
import sys
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FILLER_MAX_NEW = 96
ANSWER_MAX_NEW = 16
GATE_K = 8
GATE_THRESHOLD = 0.90
FIVE_DIGITS = re.compile(r"(?<!\d)\d{5}(?!\d)")


def encode(tok, msgs, device):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").input_ids.to(device)


@torch.no_grad()
def run_dialogue(model, tok, row):
    msgs = [{"role": "system", "content": row["system"]}]
    replies = []
    n = len(row["user_turns"])
    for i, user in enumerate(row["user_turns"]):
        msgs.append({"role": "user", "content": user})
        ids = encode(tok, msgs, model.device)
        max_new = ANSWER_MAX_NEW if i == n - 1 else FILLER_MAX_NEW
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
        reply = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        msgs.append({"role": "assistant", "content": reply})
        replies.append(reply)
    total_len = out.shape[1]
    final_prompt_len = ids.shape[1]
    m = FIVE_DIGITS.search(replies[-1])
    pred = m.group(0) if m else None
    return {
        "id": row["id"], "k": row["k"], "value": row["value"], "pred": pred,
        "correct": pred == row["value"], "final_prompt_len": final_prompt_len,
        "total_len": total_len, "replies": replies,
    }


def main(in_path="mt_niah.jsonl", out_path="gate1_full.jsonl"):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda").eval()

    rows = [json.loads(line) for line in open(in_path)]
    needle_L = len(tok(rows[0]["needle"], add_special_tokens=False).input_ids)
    pieces = tok.convert_ids_to_tokens(tok(" " + rows[0]["value"] + ".", add_special_tokens=False).input_ids)
    value_L = sum(any(c.isdigit() for c in p) for p in pieces)

    by_k = defaultdict(list)
    with open(out_path, "w") as f:
        for j, row in enumerate(rows):
            res = run_dialogue(model, tok, row)
            f.write(json.dumps(res) + "\n")
            by_k[res["k"]].append(res)
            if (j + 1) % 20 == 0:
                print(f"{j + 1}/{len(rows)}", flush=True)

    print(f"\nneedle sentence tokens = {needle_L}, value tokens L = {value_L} (example: {rows[0]['needle']!r})")
    print(f"{'k':>3} {'acc':>6} {'len_min':>8} {'len_mean':>9} {'len_max':>8}")
    for k in sorted(by_k):
        rs = by_k[k]
        acc = sum(r["correct"] for r in rs) / len(rs)
        lens = [r["total_len"] for r in rs]
        print(f"{k:>3} {acc:>6.3f} {min(lens):>8} {statistics.mean(lens):>9.0f} {max(lens):>8}")
        for b in (256, 512):
            frac = sum(l > b for l in lens) / len(lens)
            print(f"      fraction of dialogues longer than b={b}: {frac:.2f}")

    gate_rs = by_k.get(GATE_K)
    if not gate_rs:
        raise RuntimeError(f"no dialogues with k={GATE_K}")
    gate_acc = sum(r["correct"] for r in gate_rs) / len(gate_rs)
    verdict = "PASS -> stay on 0.5B" if gate_acc >= GATE_THRESHOLD else "FAIL -> switch to 1.5B, drop k=8"
    print(f"\nGate 1 (k={GATE_K}, >= {GATE_THRESHOLD:.0%}): {gate_acc:.3f}  {verdict}")


if __name__ == "__main__":
    main(*sys.argv[1:])
