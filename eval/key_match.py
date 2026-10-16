"""Do HF-transformers keys/values match the torchtune trace keys the agents were trained on?

For held-out trace samples: run the HF model on the same tokens, compare K and V per (layer, head),
and check that KVP rankings from HF inputs agree with rankings from trace inputs.
Also checks that kvp_scores() reproduces the agent's own forward() at arange positions.
"""
import argparse
from pathlib import Path

import safetensors.torch as st
import torch
from transformers import AutoModelForCausalLM

from evict_eval import MODEL, kvp_scores, load_agents

ap = argparse.ArgumentParser()
ap.add_argument("--trace_dir", required=True)
ap.add_argument("--agents_dir", default="/workspace/kv/ml-learning-to-evict/agents/grouped/oasst2_0_5b")
ap.add_argument("--samples", default="450,465,480,495")
args = ap.parse_args()

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
n_layers, n_kv = model.config.num_hidden_layers, model.config.num_key_value_heads
agents = load_agents(args.agents_dir, n_layers, n_kv, "cuda")
trace_dir = Path(args.trace_dir)

worst = {"k_rel": 0.0, "v_rel": 0.0, "k_cos": 1.0, "fwd_diff": 0.0, "overlap": 1.0, "rank_corr": 1.0}


def rank_corr(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))


with torch.no_grad():
    for s in map(int, args.samples.split(",")):
        sd = trace_dir / f"sample_{s:07d}"
        tokens = st.load_file(sd / "all_tokens.safetensors")["tensor"].long().cuda()
        P = int(st.load_file(sd / "prompt_ntokens.safetensors")["tensor"])
        L = tokens.shape[0]
        out = model(input_ids=tokens[None], position_ids=torch.arange(L, device="cuda")[None], use_cache=True)
        cache = out.past_key_values
        for l in range(n_layers):
            for h in range(n_kv):
                t = st.load_file(sd / f"layer_{l:06d}" / f"kv_head_{h:03d}" / "attention_tensors.safetensors")
                tk, tv = t["k"].float().cuda(), t["v"].float().cuda()
                hk, hv = cache.layers[l].keys[0, h].float(), cache.layers[l].values[0, h].float()
                if tk.shape != hk.shape or tv.shape != hv.shape:
                    raise SystemExit(f"sample {s} L{l} H{h}: shape trace k {tuple(tk.shape)} vs hf {tuple(hk.shape)}")
                k_rel = float((hk - tk).norm() / tk.norm())
                v_rel = float((hv - tv).norm() / tv.norm())
                k_cos = float(torch.nn.functional.cosine_similarity(hk, tk, dim=-1).min())

                agent = agents[l][h]
                pos = torch.arange(P, device="cuda")
                s_trace = kvp_scores(agent, tk[:P], tv[:P], pos)
                s_fwd = agent({"keys": tk[:P][None], "values": tv[:P][None], "queries": None,
                               "seq_lengths": torch.tensor([P], device="cuda")})["scores"][0]
                fwd_diff = float((s_trace - s_fwd).abs().max())
                s_hf = kvp_scores(agent, hk[:P], hv[:P], pos)
                top = max(1, min(128, P // 2))
                overlap = len(set(s_trace.topk(top).indices.tolist()) & set(s_hf.topk(top).indices.tolist())) / top
                rc = rank_corr(s_trace, s_hf)

                worst["k_rel"] = max(worst["k_rel"], k_rel)
                worst["v_rel"] = max(worst["v_rel"], v_rel)
                worst["k_cos"] = min(worst["k_cos"], k_cos)
                worst["fwd_diff"] = max(worst["fwd_diff"], fwd_diff)
                worst["overlap"] = min(worst["overlap"], overlap)
                worst["rank_corr"] = min(worst["rank_corr"], rc)
        print(f"sample {s}: L={L} P={P} | worst so far {worst}", flush=True)

checks = {
    "kvp_scores == agent.forward (max abs diff < 1e-4)": worst["fwd_diff"] < 1e-4,
    "keys rel err < 0.05": worst["k_rel"] < 0.05,
    "values rel err < 0.05": worst["v_rel"] < 0.05,
    "min per-token key cosine > 0.99": worst["k_cos"] > 0.99,
    "KVP top-k overlap HF vs trace > 0.9 on every head": worst["overlap"] > 0.9,
    "KVP rank correlation HF vs trace > 0.98 on every head": worst["rank_corr"] > 0.98,
}
for name, ok in checks.items():
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
if not all(checks.values()):
    raise SystemExit("KEY MATCH FAILED: do not run KVP evals; report these numbers")
print("KEY MATCH PASS")
