import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from mt_niah import CITY
from offload import choose_turn
from score_pooling import pool_scores

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_SINK = 4
DEFAULT_AGENTS = "/workspace/kv/ml-learning-to-evict/agents/grouped/oasst2_0_5b"
N_RECENT = 16
FILLER_MAX_NEW = 96
ANSWER_MAX_NEW = 16
WORDS_ANSWER_MAX_NEW = 40
FIVE_DIGITS = re.compile(r"(?<!\d)\d{5}(?!\d)")


def find_ckpt(agents_dir, layer, head):
    hits = sorted((Path(agents_dir) / f"layer_{layer:06d}" / f"kv_head_{head:03d}").glob("**/best_ckpt.pth"))
    if len(hits) != 1:
        raise FileNotFoundError(f"layer {layer} head {head}: expected 1 best_ckpt.pth, found {len(hits)}: {hits}")
    return hits[0]


def instantiate_agent_light(checkpoint_path, model_key):
    """Same steps as kvcompression.utils.utils.instantiate_agent_from_ckpt, without torchtune: import the
    checkpoint's agent class (_component_) and call it with the stored config plus device="cpu"."""
    import importlib

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = dict(ckpt["config"]["agent"])
    module, cls = cfg.pop("_component_").rsplit(".", 1)
    agent = getattr(importlib.import_module(module), cls)(**cfg, device="cpu")
    state = ckpt[model_key]
    if any(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items() if k.startswith("module.")}
    agent.load_state_dict(state, strict=True)
    return agent


def load_agents(agents_dir, n_layers, n_kv, device):
    """KVP_AGENT_LOADER=light uses instantiate_agent_light (no torchtune, e.g. on Kaggle); default is Apple's loader."""
    import os

    loader = os.environ.get("KVP_AGENT_LOADER", "apple")
    if loader == "light":
        instantiate_agent_from_ckpt = instantiate_agent_light
    elif loader == "apple":
        from kvcompression.utils.utils import instantiate_agent_from_ckpt
    else:
        raise ValueError(f"KVP_AGENT_LOADER must be 'apple' or 'light', got {loader!r}")

    agents = []
    for l in range(n_layers):
        row = []
        for h in range(n_kv):
            agent = instantiate_agent_from_ckpt(checkpoint_path=find_ckpt(agents_dir, l, h),
                                                model_key="ema_model_state_dict")
            if agent.use_kv_queries or agent.use_keydiff_features:
                raise ValueError(f"layer {l} head {h}: agent uses queries/keydiff, scorer assumes k, v, pos only")
            row.append(agent.to(device).eval())
        agents.append(row)
    return agents


def kvp_scores(agent, keys, values, positions):
    """Same features as KVSamplingGroupedQueryAgent.forward, but with original positions instead of arange."""
    k = keys.float()[None]
    v = values.float()[None]
    pos = positions.long().clamp(max=agent.max_seq_length - 1)[None]
    feats = torch.cat([
        agent.norm_key(agent.key_projection(k)),
        agent.norm_value(agent.value_projection(v)),
        agent.norm_pos(agent.pos_embedding(pos)),
    ], dim=-1)
    return agent.mlp(feats).squeeze(-1)[0]


def score_streaming(state, layer):
    return state.pos[layer].float()


def score_h2o(state, layer):
    return state.acc[layer]


def agent_positions(state, pos):
    """Positions fed to the KVP agent. With pos_scale_to=T and more than T tokens seen so far,
    original positions are linearly rescaled into [0, T-1]; the LLM itself always uses true positions."""
    if not state.pos_scale_to or state.next_pos <= state.pos_scale_to:
        return pos
    return torch.round(pos.float() * (state.pos_scale_to - 1) / (state.next_pos - 1)).long()


def score_kvp(state, layer):
    ks = state.cache.layers[layer].keys[0]
    vs = state.cache.layers[layer].values[0]
    return torch.stack([kvp_scores(state.agents[layer][h], ks[h], vs[h], agent_positions(state, state.pos[layer][h]))
                        for h in range(state.n_kv)])


def score_tova(state, layer):
    return state.last[layer]


def head_subset(row_id, n_layers, n_kv, n):
    """[n_layers, n_kv] bool mask of n heads: the first n of one fixed permutation per dialogue (nested in n)."""
    total = n_layers * n_kv
    if not 0 < n <= total:
        raise ValueError(f"protect_heads must be in 1..{total}, got {n}")
    order = random.Random(f"protect-heads-{row_id}").sample(range(total), total)
    mask = torch.zeros(total, dtype=torch.bool)
    mask[order[:n]] = True
    return mask.view(n_layers, n_kv)


SCORERS = {"streaming": score_streaming, "h2o": score_h2o, "h2o_std": score_h2o, "kvp": score_kvp,
           "tova": score_tova}
KVP_METHODS = ("kvp",)
PROTECT_MODES = ("", "value", "turn", "keyvalue")
OFFLOAD_METHODS = ("kvp", "streaming")
ACC_METHODS = ("h2o", "h2o_std")
ATTN_METHODS = ACC_METHODS + ("tova",)
POOLABLE = ("kvp", "h2o", "h2o_std", "tova")


def n_recent(method, budget):
    """Recent rows always kept. h2o_std = original H2O split: half the budget is a recent window."""
    return max(N_RECENT, budget // 2) if method == "h2o_std" else N_RECENT


class EvictingState:
    """Per-head physical eviction over an HF DynamicCache.

    pos[l] holds each cached row's original (absolute) position, shape [n_kv, n].
    New tokens get position_ids from a running counter (RoPE at true positions) and
    cache_position in cache-index space (bottom-right causal mask over kept rows).
    """

    def __init__(self, model, method, budget, record=False, agents=None, pos_scale_to=0, pool_window=0,
                 protect_needle="", protect_first=0, protect_heads=None, offload=False):
        if method != "full" and budget <= max(N_SINK, protect_first) + n_recent(method, budget):
            raise ValueError(f"budget {budget} must exceed "
                             f"{max(N_SINK, protect_first) + n_recent(method, budget)} protected rows")
        if protect_first and method == "full":
            raise ValueError("protect_first needs an eviction method")
        if method in KVP_METHODS and agents is None:
            raise ValueError(f"method {method} needs agents")
        if pos_scale_to and method != "kvp":
            raise ValueError("pos_scale_to only applies to kvp")
        if pool_window and method not in POOLABLE:
            raise ValueError(f"pool_window applies to {POOLABLE}, not {method}")
        if protect_needle not in PROTECT_MODES:
            raise ValueError(f"protect_needle must be one of {PROTECT_MODES}, got {protect_needle!r}")
        if protect_needle and method == "full":
            raise ValueError("protect_needle needs an eviction method")
        if protect_heads is not None and not protect_needle:
            raise ValueError("protect_heads needs protect_needle")
        if offload and (method not in OFFLOAD_METHODS or protect_needle or pool_window):
            raise ValueError(f"offload_retrieve needs a method in {OFFLOAD_METHODS}, without protect_needle or pool_window")
        self.offload = offload
        self.arch_k = [None] * model.config.num_hidden_layers
        self.arch_v = [None] * model.config.num_hidden_layers
        self.n_restored = 0
        self.protect_heads = None if protect_heads is None else protect_heads.to(model.device)
        self.agents = agents
        self.pos_scale_to = pos_scale_to
        self.pool_window = pool_window
        self.protect_needle = protect_needle
        self.protect_first = protect_first
        self.force_pos = None
        self.n_forced = 0
        self.pool_changed = []
        self.model = model
        self.method = method
        self.budget = budget
        self.n_layers = model.config.num_hidden_layers
        self.n_kv = model.config.num_key_value_heads
        self.group = model.config.num_attention_heads // self.n_kv
        self.cache = DynamicCache()
        self.pos = [None] * self.n_layers
        self.acc = [None] * self.n_layers
        self.last = [None] * self.n_layers
        self.next_pos = 0
        self.record = record
        self.visible_log = []

    def cache_len(self):
        return 0 if self.pos[0] is None else self.pos[0].shape[1]

    @torch.no_grad()
    def forward(self, ids):
        dev = self.model.device
        n_new = len(ids)
        new_pos = torch.arange(self.next_pos, self.next_pos + n_new, device=dev)
        cache_pos = torch.arange(self.cache_len(), self.cache_len() + n_new, device=dev)
        if self.record:
            before = None
            if self.pos[0] is not None:
                ref = self.pos[0][0]
                for p in self.pos:
                    if not torch.equal(p, ref.expand_as(p)):
                        raise RuntimeError("record mode needs identical kept sets across heads/layers")
                before = ref.tolist()
            self.visible_log.append((before, new_pos.tolist()))
        want_attn = self.method in ATTN_METHODS
        out = self.model(
            input_ids=torch.tensor([ids], device=dev),
            past_key_values=self.cache,
            position_ids=new_pos[None],
            cache_position=cache_pos,
            use_cache=True,
            output_attentions=want_attn,
            logits_to_keep=1,
        )
        self.cache = out.past_key_values
        if self.offload:
            for l in range(self.n_layers):
                k = self.cache.layers[l].keys[0, :, -n_new:].cpu()
                v = self.cache.layers[l].values[0, :, -n_new:].cpu()
                self.arch_k[l] = k if self.arch_k[l] is None else torch.cat([self.arch_k[l], k], dim=1)
                self.arch_v[l] = v if self.arch_v[l] is None else torch.cat([self.arch_v[l], v], dim=1)
                if self.arch_k[l].shape[1] != self.next_pos + n_new:
                    raise RuntimeError(f"layer {l}: archive holds {self.arch_k[l].shape[1]} rows, expected {self.next_pos + n_new}")
        for l in range(self.n_layers):
            p = new_pos.expand(self.n_kv, -1)
            self.pos[l] = p.clone() if self.pos[l] is None else torch.cat([self.pos[l], p], dim=1)
            if want_attn:
                a = out.attentions[l][0].float()
                if a.shape[-1] != self.pos[l].shape[1]:
                    raise RuntimeError(f"layer {l}: attention covers {a.shape[-1]} keys, cache has {self.pos[l].shape[1]}")
                a = a.view(self.n_kv, self.group, n_new, -1)
                if self.method in ACC_METHODS:
                    prev = self.acc[l] if self.acc[l] is not None else torch.zeros(self.n_kv, 0, device=dev)
                    self.acc[l] = torch.cat([prev, torch.zeros(self.n_kv, n_new, device=dev)], dim=1) + a.sum(dim=(1, 2))
                else:
                    self.last[l] = a[:, :, -1, :].sum(dim=1)
        self.next_pos += n_new
        return out.logits[0, -1]

    @torch.no_grad()
    def evict(self):
        n = self.cache_len()
        if self.method == "full" or n <= self.budget:
            return
        scorer = SCORERS[self.method]
        protect = torch.zeros(n, dtype=torch.bool, device=self.pos[0].device)
        protect[:N_SINK] = True
        protect[n - n_recent(self.method, self.budget):] = True
        for l in range(self.n_layers):
            layer = self.cache.layers[l]
            if layer.keys.shape[2] != n:
                raise RuntimeError(f"layer {l} has {layer.keys.shape[2]} rows, expected {n}")
            s = scorer(self, l).float()
            prot = protect.expand(self.n_kv, -1)
            if self.force_pos is not None:
                forced = (self.pos[l][:, :, None] == self.force_pos).any(-1)
                if self.protect_heads is not None:
                    forced = forced & self.protect_heads[l][:, None]
                prot = prot | forced
            if self.protect_first:
                prot = prot | (self.pos[l] < self.protect_first)
            if self.pool_window:
                raw_keep = s.masked_fill(prot, float("inf")).topk(self.budget, dim=-1).indices
                s = pool_scores(s.masked_fill(prot, s.min().item()), self.pos[l], self.pool_window)
                new_keep = s.masked_fill(prot, float("inf")).topk(self.budget, dim=-1).indices
                changed = 1.0 - (new_keep[..., None] == raw_keep[..., None, :]).any(-1).float().mean(-1)
                self.pool_changed.append(changed.mean().item())
            keep = s.masked_fill(prot, float("inf")).topk(self.budget, dim=-1).indices.sort(dim=-1).values
            idx = keep[None, :, :, None].expand(1, self.n_kv, self.budget, layer.keys.shape[-1])
            layer.keys = layer.keys.gather(2, idx).contiguous()
            layer.values = layer.values.gather(2, idx).contiguous()
            self.pos[l] = self.pos[l].gather(1, keep)
            if self.acc[l] is not None:
                self.acc[l] = self.acc[l].gather(1, keep)
            if self.last[l] is not None:
                self.last[l] = self.last[l].gather(1, keep)


    @torch.no_grad()
    def restore(self, positions):
        """Offload mode: bring `positions` back from the CPU archive into every head, at constant cache size.
        Per head, missing positions are added and the lowest-scored rows (KVP score, or position for streaming)
        that are not sinks, recent or restored
        are displaced; the restored positions are then protected (force_pos) in all heads until the next restore."""
        if not self.offload:
            raise RuntimeError("restore needs offload mode")
        dev = self.model.device
        R = torch.tensor(sorted(set(positions)), device=dev)
        if R.numel() == 0 or int(R.max()) >= self.next_pos:
            raise ValueError("restore positions must be non-empty and already seen")
        n = self.cache_len()
        recent = n_recent(self.method, self.budget)
        for l in range(self.n_layers):
            layer = self.cache.layers[l]
            ks, vs, ps = [], [], []
            for h in range(self.n_kv):
                p = self.pos[l][h]
                miss = R[~torch.isin(R, p)]
                k = torch.cat([layer.keys[0, h], self.arch_k[l][h][miss.cpu()].to(dev, layer.keys.dtype)])
                v = torch.cat([layer.values[0, h], self.arch_v[l][h][miss.cpu()].to(dev, layer.values.dtype)])
                pp = torch.cat([p, miss])
                order = pp.argsort()
                prot = torch.isin(pp, R)
                prot[order[:N_SINK]] = True
                prot[order[len(pp) - recent:]] = True
                if int(prot.sum()) > n:
                    raise RuntimeError(f"layer {l} head {h}: {int(prot.sum())} protected rows exceed cache size {n}")
                if self.method == "kvp":
                    s = kvp_scores(self.agents[l][h], k, v, agent_positions(self, pp)).float()
                else:
                    s = pp.float()
                keep = s.masked_fill(prot, float("inf")).topk(n).indices
                keep = keep[pp[keep].argsort()]
                ks.append(k[keep])
                vs.append(v[keep])
                ps.append(pp[keep])
            layer.keys = torch.stack(ks)[None].contiguous()
            layer.values = torch.stack(vs)[None].contiguous()
            self.pos[l] = torch.stack(ps)
        self.force_pos = R
        self.n_restored = int(R.numel())


def turn_tokens(tok, row, i):
    if i == 0:
        msgs = [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["user_turns"][0]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        text = "\n<|im_start|>user\n" + row["user_turns"][i] + "<|im_end|>\n<|im_start|>assistant\n"
    return tok(text, add_special_tokens=False).input_ids


def find_needle(tok, ids, value):
    pieces = [tok.decode([t]) for t in ids]
    n = len(value)
    for i in range(len(ids) - n + 1):
        window = pieces[i:i + n]
        if all(p.strip().isdigit() for p in window) and "".join(window).replace(" ", "") == value:
            return list(range(i, i + n))
    raise RuntimeError(f"needle value {value} not found as {n} digit tokens")


def find_spans(tok, ids, texts):
    """Token indices covering the first occurrence of each text in the decoded ids (word needles)."""
    pieces = [tok.decode([t]) for t in ids]
    starts, n = [], 0
    for p in pieces:
        starts.append(n)
        n += len(p)
    full = "".join(pieces)
    out = []
    for text in texts:
        at = full.find(text)
        if at == -1:
            raise RuntimeError(f"needle text {text!r} not found in the needle turn")
        end = at + len(text)
        out += [i for i, s in enumerate(starts) if s < end and s + len(pieces[i]) > at]
    return sorted(set(out))


def needle_positions(tok, ids, row):
    if row.get("needle_type", "digits") == "words":
        return find_spans(tok, ids, [row["name"], row["city"]])
    return find_needle(tok, ids, row["value"])


def key_positions(tok, ids, row, start):
    """Positions of the linking phrase (row["key"], e.g. "cousin's hamster" or "silent harbor C6") in the needle turn."""
    if row["key"] not in row["needle"]:
        raise RuntimeError(f"key {row['key']!r} not in needle {row['needle']!r}")
    return [start + j for j in find_spans(tok, ids, [row["key"]])]


def score(row, answer):
    """Returns (pred, correct, extra fields for the record)."""
    if row.get("needle_type", "digits") == "words":
        a = answer.lower()
        hedged = any(c.lower() in a for c in CITY if c != row["city"])
        name_ok, city_ok = row["name"].lower() in a, row["city"].lower() in a and not hedged
        return answer.strip(), name_ok and city_ok, {"name_ok": name_ok, "city_ok": city_ok, "city_hedged": hedged}
    m = FIVE_DIGITS.search(answer)
    pred = m.group(0) if m else None
    return pred, pred == row["value"], {}


def survival(state, positions):
    """Share of the given positions still cached, per (layer, head): [n_layers, n_kv]. NaN before the needle exists."""
    if positions is None:
        return np.full((state.n_layers, state.n_kv), np.nan, dtype=np.float32)
    t = torch.tensor(positions, device=state.pos[0].device)
    return np.stack([(p[:, :, None] == t).any(dim=1).float().mean(dim=1).cpu().numpy() for p in state.pos])


def run_dialogue(state, tok, row, eos_ids, im_end, trace=None):
    n_turns = len(row["user_turns"])
    needle_turn = row.get("needle_turn", 0)
    sys_len = len(tok(tok.apply_chat_template([{"role": "system", "content": row["system"]}], tokenize=False),
                      add_special_tokens=False).input_ids)
    events, kept, needle_s, sys_s = [], [], [], []
    pending, stream, needle = [], [], None

    def log_event(label):
        events.append(label)
        needle_s.append(survival(state, needle))
        sys_s.append(survival(state, list(range(sys_len))))
        if state.method != "full":
            k = np.stack([p.cpu().numpy() for p in state.pos]).astype(np.int32)
            if k.shape[2] > state.budget:
                raise RuntimeError(f"logged {k.shape[2]} rows, budget {state.budget}")
            padded = np.full(k.shape[:2] + (state.budget,), -1, dtype=np.int32)
            padded[..., :k.shape[2]] = k
            kept.append(padded)

    user_ranges, retrieval = [], {}
    for i in range(n_turns):
        tt = turn_tokens(tok, row, i)
        start = state.next_pos + len(pending)
        if state.offload and i > 0:
            j, sc = choose_turn(row["user_turns"][:i], row["user_turns"][i])
            if j is not None:
                state.restore(user_ranges[j])
            else:
                state.force_pos, state.n_restored = None, 0
            retrieval = {"retrieved_turn": j, "retrieval_score": sc, "n_restored": state.n_restored if j is not None else 0}
        user_ranges.append(list(range(sys_len, len(tt)) if i == 0 else range(start, start + len(tt))))
        if i == needle_turn:
            needle_tt, start_of_needle = tt, start
            needle = [start + j for j in needle_positions(tok, tt, row)]
            if state.protect_needle:
                turn = range(sys_len, len(tt)) if i == 0 else range(start, start + len(tt))
                if state.protect_needle == "value":
                    force = needle
                elif state.protect_needle == "keyvalue":
                    force = sorted(set(needle) | set(key_positions(tok, tt, row, start)))
                else:
                    force = list(turn)
                state.n_forced = len(force)
                state.force_pos = torch.tensor(force, device=state.model.device)
        ids = pending + tt
        logits = state.forward(ids)
        stream += ids
        if trace is not None:
            trace.append((state.next_pos - 1, logits.float().cpu()))
        state.evict()
        log_event(f"t{i}_prefill")
        answer_max = WORDS_ANSWER_MAX_NEW if row.get("needle_type") == "words" else ANSWER_MAX_NEW
        max_new = answer_max if i == n_turns - 1 else FILLER_MAX_NEW
        reply = [int(logits.argmax())]
        while reply[-1] not in eos_ids and len(reply) < max_new:
            logits = state.forward([reply[-1]])
            stream.append(reply[-1])
            if trace is not None:
                trace.append((state.next_pos - 1, logits.float().cpu()))
            reply.append(int(logits.argmax()))
        if i < n_turns - 1:
            state.evict()
            log_event(f"t{i}_end")
        pending = [im_end] if reply[-1] in eos_ids else [reply[-1], im_end]

    answer = tok.decode(reply, skip_special_tokens=True)
    pred, correct, extra = score(row, answer)
    if row.get("needle_type") == "words":
        for part in ("name", "city"):
            pos = [start_of_needle + j for j in find_spans(tok, needle_tt, [row[part]])]
            extra[f"{part}_heads_full_at_answer"] = float((survival(state, pos) == 1.0).mean())
    if state.protect_needle:
        extra["n_forced"] = state.n_forced
    if state.offload:
        extra.update(retrieval)
        extra["retrieval_hit"] = retrieval.get("retrieved_turn") == needle_turn
    return {
        "answer": answer, "pred": pred, "correct": correct, "extra": extra, "n_needle": len(needle), "stream": stream,
        "events": events, "needle": np.stack(needle_s), "sys": np.stack(sys_s),
        "kept": np.stack(kept) if kept else None,
    }


def load_model(dtype, model_name=MODEL):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, attn_implementation="eager").to("cuda").eval()
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) | {im_end}
    return tok, model, eos_ids, im_end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["full", "streaming", "h2o", "h2o_std", "tova", "kvp"])
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--pos_scale_to", type=int, default=0,
                    help="kvp only: rescale agent-input positions into [0, T-1] once more than T tokens are seen")
    ap.add_argument("--tag", default="", help="suffix for the method name in outputs, e.g. pos600")
    ap.add_argument("--pool_window", type=int, default=0,
                    help="max-pool scores over neighbours within w original positions (kvp/h2o/h2o_std/tova)")
    ap.add_argument("--protect_needle", default="", choices=list(PROTECT_MODES),
                    help="never evict the needle value tokens (value), value + linking phrase row['key'] (keyvalue), "
                         "or the whole needle user turn (turn)")
    ap.add_argument("--protect_heads", type=int, default=0,
                    help="with --protect_needle: force-keep the needle only in N random (layer, head) pairs per "
                         "dialogue (nested: the N heads are the first N of one fixed permutation per dialogue id)")
    ap.add_argument("--model", default=MODEL, help="HF model id (batch v8: Qwen/Qwen2.5-1.5B-Instruct)")
    ap.add_argument("--offload_retrieve", action="store_true",
                    help="kvp only: keep evicted K/V in a CPU archive; at each new user turn restore the best-matching "
                         "earlier user turn (offload.choose_turn) into all heads at constant cache size (batch v7)")
    ap.add_argument("--protect_first", type=int, default=0,
                    help="baseline/defence check: never evict the first N positions (system prompt + start of turn 1)")
    ap.add_argument("--agents_dir", default=DEFAULT_AGENTS)
    ap.add_argument("--data", default="/workspace/kv/mt_niah.jsonl")
    ap.add_argument("--out_dir", default="/workspace/kv/logs")
    ap.add_argument("--ks", default="2,4,8")
    ap.add_argument("--limit", type=int, default=0, help="max dialogues per k (0 = all)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    tok, model, eos_ids, im_end = load_model(dtype, args.model)
    agents = None
    if args.method in KVP_METHODS:
        agents = load_agents(args.agents_dir, model.config.num_hidden_layers,
                             model.config.num_key_value_heads, model.device)
    if args.protect_heads and not args.protect_needle:
        raise ValueError("--protect_heads needs --protect_needle")
    if (args.pos_scale_to or args.pool_window or args.protect_needle or args.protect_first or args.offload_retrieve) \
            and not args.tag:
        raise ValueError("--pos_scale_to/--pool_window/--protect_needle/--protect_first need a --tag "
                         "so results do not overwrite the base run")
    if args.agents_dir != DEFAULT_AGENTS and args.method in KVP_METHODS and not args.tag:
        raise ValueError("a non-default --agents_dir needs a --tag (e.g. ft) so results do not overwrite the base run")
    label = f"{args.method}_{args.tag}" if args.tag else args.method
    name = f"{label}_b{args.budget}_{args.dtype}"
    out_dir = Path(args.out_dir)
    (out_dir / name).mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.open()}

    ks = {int(k) for k in args.ks.split(",")}
    rows = [json.loads(l) for l in open(args.data)]
    rows = [r for r in rows if r["k"] in ks]
    if args.limit:
        seen, limited = {}, []
        for r in rows:
            seen[r["k"]] = seen.get(r["k"], 0) + 1
            if seen[r["k"]] <= args.limit:
                limited.append(r)
        rows = limited

    with out_path.open("a") as f:
        for j, row in enumerate(rows):
            if row["id"] in done:
                continue
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            heads = (head_subset(row["id"], model.config.num_hidden_layers, model.config.num_key_value_heads,
                                 args.protect_heads) if args.protect_heads else None)
            state = EvictingState(model, args.method, args.budget, agents=agents, pos_scale_to=args.pos_scale_to,
                                  pool_window=args.pool_window, protect_needle=args.protect_needle,
                                  protect_first=args.protect_first, protect_heads=heads,
                                  offload=args.offload_retrieve)
            res = run_dialogue(state, tok, row, eos_ids, im_end)
            np.savez_compressed(out_dir / name / f"{row['id']}.npz", events=np.array(res["events"]),
                                needle=res["needle"], sys=res["sys"],
                                stream=np.array(res["stream"], dtype=np.int32),
                                **({"kept": res["kept"]} if res["kept"] is not None else {}))
            rec = {
                "id": row["id"], "k": row["k"], "method": label, "budget": args.budget,
                "pos_scale_to": args.pos_scale_to,
                "pool_window": args.pool_window, "protect_needle": args.protect_needle,
                "protect_first": args.protect_first, "protect_heads": args.protect_heads,
                "offload_retrieve": args.offload_retrieve, "model": args.model,
                "lead": row.get("needle_turn", 0),
                "agents_dir": args.agents_dir if args.method in KVP_METHODS else None,
                "pool_changed": (sum(state.pool_changed) / len(state.pool_changed)) if state.pool_changed else None,
                "value": row["value"], "pred": res["pred"], "correct": res["correct"], "answer": res["answer"],
                "needle_type": row.get("needle_type", "digits"), "n_needle_tokens": int(res["n_needle"]), **res["extra"],
                "total_len": state.next_pos,
                "needle_at_answer": float(res["needle"][-1].mean()),
                "needle_heads_full_at_answer": float((res["needle"][-1] == 1.0).mean()),
                "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20,
                "seconds": time.time() - t0,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (j + 1) % 20 == 0:
                print(f"{j + 1}/{len(rows)}", flush=True)

    recs = [json.loads(l) for l in out_path.open()]
    print(f"\n{name}")
    for k in sorted(ks):
        rs = [r for r in recs if r["k"] == k]
        if rs:
            acc = sum(r["correct"] for r in rs) / len(rs)
            surv = sum(r["needle_at_answer"] for r in rs) / len(rs)
            print(f"k={k}: acc {acc:.3f} (n={len(rs)}), needle survival at answer {surv:.3f}")


if __name__ == "__main__":
    main()
