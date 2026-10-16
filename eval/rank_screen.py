"""Offline screen: where do KVP agents rank the needle digits among all tokens, before any eviction?

KVP scores depend only on (key, value, original position), so per-turn eviction is close to one global top-b cut
over the whole dialogue. This script runs each MT-NIAH dialogue with the FULL cache up to the question, scores every
cached token with each agent set, and reports for the 5 code digits:
  pct      mean percentile rank among evictable tokens (1.0 = scored above every other token, 0.5 = average)
  kept@b   share of (layer, head, digit) predicted to survive a single top-b cut (b minus the always-kept rows)
  all5@b   share of (layer, head) predicted to keep all 5 digits
Approximation: under real physical eviction later keys are computed over an already-evicted context, so compare
kept@b for the original agents with the measured needle_at_answer of the real runs before trusting it.

  python eval/rank_screen.py --data results/mt_niah.jsonl --budgets 128,256 \
      --agents base=$A ft=$P/agents/grouped/oasst2_recall_ft_0_5b --out results/rank_screen.md
"""
import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from evict_eval import (FILLER_MAX_NEW, N_RECENT, N_SINK, SCORERS, EvictingState, find_needle, load_agents,
                        load_model, turn_tokens)


@torch.no_grad()
def full_cache_until_question(model, tok, row, eos_ids, im_end):
    state = EvictingState(model, "full", 0)
    needle_turn = row.get("needle_turn", 0)
    pending, needle = [], None
    n_turns = len(row["user_turns"])
    for i in range(n_turns):
        tt = turn_tokens(tok, row, i)
        if i == needle_turn:
            start = state.next_pos + len(pending)
            needle = [start + j for j in find_needle(tok, tt, row["value"])]
        logits = state.forward(pending + tt)
        if i == n_turns - 1:
            break
        reply = [int(logits.argmax())]
        while reply[-1] not in eos_ids and len(reply) < FILLER_MAX_NEW:
            logits = state.forward([reply[-1]])
            reply.append(int(logits.argmax()))
        pending = [im_end] if reply[-1] in eos_ids else [reply[-1], im_end]
    return state, needle


@torch.no_grad()
def needle_ranks(state, needle, agents):
    """Per (layer, head, digit): number of evictable tokens scored strictly higher, and the evictable count."""
    state.agents = agents
    n = state.cache_len()
    evictable = torch.ones(n, dtype=torch.bool, device=state.pos[0].device)
    evictable[:N_SINK] = False
    evictable[n - N_RECENT:] = False
    idx = torch.tensor(needle, device=evictable.device)
    if not evictable[idx].all():
        raise RuntimeError("needle overlaps the always-kept rows; screen not meaningful")
    above = []
    for l in range(state.n_layers):
        s = SCORERS["kvp"](state, l).float()
        if not torch.equal(state.pos[l][0], torch.arange(n, device=s.device)):
            raise RuntimeError("full cache rows are not positions 0..n-1")
        ev = s[:, evictable]
        above.append(torch.stack([(ev > s[:, d:d + 1]).sum(-1) for d in needle], dim=-1))
    return torch.stack(above).cpu().numpy(), int(evictable.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--agents", nargs="+", required=True, help="label=agents_dir ...")
    ap.add_argument("--budgets", default="128,256")
    ap.add_argument("--ks", default="2,4,8")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok, model, eos_ids, im_end = load_model(torch.bfloat16)
    sets = {}
    for spec in args.agents:
        label, path = spec.split("=", 1)
        sets[label] = load_agents(path, model.config.num_hidden_layers, model.config.num_key_value_heads,
                                  model.device)
    budgets = [int(b) for b in args.budgets.split(",")]
    ks = {int(k) for k in args.ks.split(",")}
    rows = [r for r in map(json.loads, open(args.data)) if r["k"] in ks]
    if args.limit:
        count, limited = defaultdict(int), []
        for r in rows:
            count[r["k"]] += 1
            if count[r["k"]] <= args.limit:
                limited.append(r)
        rows = limited

    stats = defaultdict(lambda: defaultdict(list))
    for j, row in enumerate(rows):
        state, needle = full_cache_until_question(model, tok, row, eos_ids, im_end)
        for label, agents in sets.items():
            above, n_ev = needle_ranks(state, needle, agents)
            st = stats[(label, row["k"])]
            st["pct"].append(float((1.0 - above / max(n_ev - 1, 1)).mean()))
            st["len"].append(state.cache_len())
            for b in budgets:
                slots = b - N_SINK - N_RECENT
                kept = above < slots
                st[f"kept@{b}"].append(float(kept.mean()))
                st[f"all5@{b}"].append(float(kept.all(-1).mean()))
        if (j + 1) % 20 == 0:
            print(f"{j + 1}/{len(rows)}", flush=True)

    cols = ["pct"] + [f"{m}@{b}" for b in budgets for m in ("kept", "all5")]
    lines = ["# Needle rank screen (full cache up to the question, single-cut approximation)", "",
             f"data: {args.data}; agents: {', '.join(args.agents)}", "",
             "| agents | k | n | mean len | " + " | ".join(cols) + " |",
             "|---|---|---|---|" + "---|" * len(cols)]
    for (label, k) in sorted(stats):
        st = stats[(label, k)]
        lines.append(f"| {label} | {k} | {len(st['pct'])} | {np.mean(st['len']):.0f} | "
                     + " | ".join(f"{np.mean(st[c]):.3f}" for c in cols) + " |")
    text = "\n".join(lines) + "\n"
    open(args.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
