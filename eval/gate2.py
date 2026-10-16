"""Gate 2: per-head ranking cost of KVP vs StreamingLLM on held-out OASST2 traces.

Cost is the repo's own oracle (Oracle.compute_cost): future-attention AUC of the ranking,
normalized by the ideal ranking, so 1.0 = perfect and lower is better (training reward = -cost).
Cut point = natural prompt length; the future is the model's own reply.
"""
import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("DISABLE_TORCH_COMPILER", "1")

import safetensors.torch as st  # noqa: E402
import torch  # noqa: E402

from evict_eval import N_RECENT, N_SINK, kvp_scores, load_agents  # noqa: E402
from kvcompression.rl.rewards.oracle import Oracle  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--trace_dir", required=True)
ap.add_argument("--agents_dir", default="/workspace/kv/ml-learning-to-evict/agents/grouped/oasst2_0_5b")
ap.add_argument("--split", default="test.txt")
ap.add_argument("--out", default="/workspace/kv/gate2_per_head.csv")
ap.add_argument("--n_layers", type=int, default=24)
ap.add_argument("--n_kv", type=int, default=2)
args = ap.parse_args()

trace_dir = Path(args.trace_dir)
samples = [s.strip() for s in (trace_dir / args.split).read_text().splitlines() if s.strip()]
agents = load_agents(args.agents_dir, args.n_layers, args.n_kv, "cuda")
oracle = Oracle()
METHODS = ["kvp", "kvp_raw", "streaming", "h2o_prefix", "random"]


def protected_order(scores, P):
    s = scores.clone().float()
    s[:min(N_SINK, P)] = float("inf")
    s[max(0, P - N_RECENT):] = float("inf")
    return torch.argsort(s, descending=True, stable=True)


def cost(rank, k, q, P, L):
    r = oracle.compute_cost(kv_rankings=rank[None], kv_rankings_length=torch.tensor([P], device="cuda"),
                            all_keys=k[None], all_queries=q[None], seq_lengths=torch.tensor([L], device="cuda"))
    c = float(r.cost.reshape(-1)[0])
    if not math.isfinite(c) or c < 1 - 1e-4:
        raise RuntimeError(f"invalid cost {c}")
    return c


per_head = {(l, h): {m: [] for m in METHODS} for l in range(args.n_layers) for h in range(args.n_kv)}
g = torch.Generator(device="cpu").manual_seed(0)

with torch.no_grad():
    for i, name in enumerate(samples):
        sd = trace_dir / name
        P = int(st.load_file(sd / "prompt_ntokens.safetensors")["tensor"])
        for l in range(args.n_layers):
            for h in range(args.n_kv):
                t = st.load_file(sd / f"layer_{l:06d}" / f"kv_head_{h:03d}" / "attention_tensors.safetensors")
                k, v, q = t["k"].float().cuda(), t["v"].float().cuda(), t["q_group"].float().cuda()
                L = k.shape[0]
                if P >= L:
                    raise RuntimeError(f"{name}: no future tokens (P={P}, L={L})")
                pos = torch.arange(P, device="cuda")
                s_kvp = kvp_scores(agents[l][h], k[:P], v[:P], pos)
                att = torch.softmax((q[:, :P] @ k[:P].T) / math.sqrt(k.shape[-1])
                                    + torch.triu(torch.full((P, P), float("-inf"), device="cuda"), 1), dim=-1)
                s_h2o = att.sum(dim=(0, 1))
                ranks = {
                    "kvp": protected_order(s_kvp, P),
                    "kvp_raw": torch.argsort(s_kvp, descending=True, stable=True),
                    "streaming": protected_order(pos.float(), P),
                    "h2o_prefix": protected_order(s_h2o, P),
                    "random": torch.randperm(P, generator=g).cuda(),
                }
                for m, r in ranks.items():
                    per_head[(l, h)][m].append(cost(r, k, q, P, L))
        print(f"{i + 1}/{len(samples)} {name} P={P}", flush=True)

rows, wins = [], 0
for (l, h), d in sorted(per_head.items()):
    mean = {m: sum(v) / len(v) for m, v in d.items()}
    paired = sum(a < b for a, b in zip(d["kvp"], d["streaming"])) / len(d["kvp"])
    beats = mean["kvp"] < mean["streaming"]
    wins += beats
    rows.append({"layer": l, "head": h, **{f"cost_{m}": round(mean[m], 4) for m in METHODS},
                 "kvp_paired_winrate_vs_streaming": round(paired, 3), "kvp_beats_streaming": beats})

with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

n_heads = len(rows)
print(f"\n{'L':>3} {'H':>2} " + " ".join(f"{m:>11}" for m in METHODS) + "  winrate")
for r in rows:
    print(f"{r['layer']:>3} {r['head']:>2} " + " ".join(f"{r['cost_' + m]:>11.3f}" for m in METHODS)
          + f"  {r['kvp_paired_winrate_vs_streaming']:.2f}")
avg = {m: sum(r[f"cost_{m}"] for r in rows) / n_heads for m in METHODS}
print("\nmean cost over heads: " + ", ".join(f"{m} {avg[m]:.3f}" for m in METHODS))
print(f"heads where KVP (protected) beats StreamingLLM: {wins}/{n_heads}")
print(f"heads where KVP beats H2O-prefix: {sum(r['cost_kvp'] < r['cost_h2o_prefix'] for r in rows)}/{n_heads}")
print(f"heads where KVP beats random: {sum(r['cost_kvp_raw'] < r['cost_random'] for r in rows)}/{n_heads}")
print(f"\nGate 2: {'PASS' if wins > n_heads / 2 else 'FAIL'} (KVP below StreamingLLM on most heads)")
