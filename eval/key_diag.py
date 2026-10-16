"""Is the HF-vs-trace K/V mismatch ordinary bf16 noise, or a localized difference?

Noise floor = HF bf16 vs HF fp32 on the same tokens (same framework, only rounding differs).
Pass rule (fixed before looking at results):
  1. per layer, the worst trace-vs-HF rel error over heads/samples <= 2x the worst bf16-vs-fp32 rel error, for K and V
  2. no (layer, head) is an outlier: its worst V rel error <= 3x the median over all heads
"""
import argparse
import statistics
from collections import defaultdict
from pathlib import Path

import safetensors.torch as st
import torch
from transformers import AutoModelForCausalLM

from evict_eval import MODEL

ap = argparse.ArgumentParser()
ap.add_argument("--trace_dir", required=True)
ap.add_argument("--samples", default="450,465,480,495")
args = ap.parse_args()

m16 = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
m32 = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, attn_implementation="eager").cuda().eval()
n_layers, n_kv = m16.config.num_hidden_layers, m16.config.num_key_value_heads


def rel(a, b):
    return float((a - b).norm() / b.norm())


trace = defaultdict(lambda: [0.0, 0.0, 1.0])
floor = defaultdict(lambda: [0.0, 0.0])
with torch.no_grad():
    for s in map(int, args.samples.split(",")):
        sd = Path(args.trace_dir) / f"sample_{s:07d}"
        tokens = st.load_file(sd / "all_tokens.safetensors")["tensor"].long().cuda()[None]
        pos = torch.arange(tokens.shape[1], device="cuda")[None]
        c16 = m16(input_ids=tokens, position_ids=pos, use_cache=True).past_key_values
        c32 = m32(input_ids=tokens, position_ids=pos, use_cache=True).past_key_values
        for l in range(n_layers):
            for h in range(n_kv):
                t = st.load_file(sd / f"layer_{l:06d}" / f"kv_head_{h:03d}" / "attention_tensors.safetensors")
                tk, tv = t["k"].float().cuda(), t["v"].float().cuda()
                hk, hv = c16.layers[l].keys[0, h].float(), c16.layers[l].values[0, h].float()
                fk, fv = c32.layers[l].keys[0, h], c32.layers[l].values[0, h]
                cos = torch.nn.functional.cosine_similarity(hk, tk, dim=-1)
                tr = trace[(l, h)]
                tr[0] = max(tr[0], rel(hk, tk))
                tr[1] = max(tr[1], rel(hv, tv))
                tr[2] = min(tr[2], float(torch.quantile(cos, 0.01)))
                fl = floor[(l, h)]
                fl[0] = max(fl[0], rel(hk, fk))
                fl[1] = max(fl[1], rel(hv, fv))
        print(f"sample {s} done", flush=True)

print(f"\n{'layer':>5} {'K trace':>8} {'K floor':>8} {'V trace':>8} {'V floor':>8} {'cos p1':>7}  ok")
layer_ok = True
for l in range(n_layers):
    kt = max(trace[(l, h)][0] for h in range(n_kv))
    vt = max(trace[(l, h)][1] for h in range(n_kv))
    kf = max(floor[(l, h)][0] for h in range(n_kv))
    vf = max(floor[(l, h)][1] for h in range(n_kv))
    cp = min(trace[(l, h)][2] for h in range(n_kv))
    ok = kt <= 2 * kf and vt <= 2 * vf
    layer_ok &= ok
    print(f"{l:>5} {kt:>8.4f} {kf:>8.4f} {vt:>8.4f} {vf:>8.4f} {cp:>7.4f}  {'ok' if ok else 'ABOVE 2x FLOOR'}")

v_all = [trace[k][1] for k in trace]
med = statistics.median(v_all)
outliers = [(l, h, round(trace[(l, h)][1], 4)) for (l, h) in sorted(trace) if trace[(l, h)][1] > 3 * med]
print(f"\nmedian worst-V rel error over heads: {med:.4f}; outlier heads (> 3x median): {outliers or 'none'}")
print("rule 1 (trace error within 2x bf16 noise floor, every layer):", "PASS" if layer_ok else "FAIL")
print("rule 2 (no outlier head):", "PASS" if not outliers else "FAIL")
if layer_ok and not outliers:
    print("DIAGNOSIS: ordinary bf16 noise -> OK to run KVP evals")
else:
    raise SystemExit("DIAGNOSIS: mismatch beyond bf16 noise -> do not run KVP evals; report the table")
