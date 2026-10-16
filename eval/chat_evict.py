"""Chat live with Qwen2.5-0.5B-Instruct while its KV cache is evicted after every turn.

Runs on CPU (fp32) or GPU (bf16 where supported, else fp32, e.g. Kaggle T4). Each method keeps its own copy of
the conversation, so you can compare, e.g., KVP against the full cache on the same questions.

  python chat_evict.py --methods kvp,kvp+fix,full --budget 256 --agents_dir path/to/agents
  python chat_evict.py --methods kvp,kvp+pool4,kvp+pool8,full --budget 128 --agents_dir path/to/agents

+fix (kvp or streaming, batch v7): every evicted token is kept in CPU memory; when you send a message, the earlier
message of yours that best matches it is restored whole into all heads (same GPU cache size).
--retriever embed (default, sentence embeddings) or lexical (shared words, the one used in the GPU experiments).

Commands:
  /memory [method]   show the conversation as the cache sees it (kept by all / some / no heads)
  /find TEXT         per-token share of heads holding TEXT, in every method (e.g. /find Biscuit)
  /reset             start a new conversation
  /quit
Needs: torch, transformers==4.55.4, numpy (+ omegaconf if the checkpoints need it).
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import transformers
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evict_eval import MODEL, N_RECENT, N_SINK, EvictingState  # noqa: E402
from offload import choose_turn  # noqa: E402

SYSTEM = "You are a helpful assistant."
RESET, BOLD, DIM, YELLOW, RED, GREY = "\033[0m", "\033[1m", "\033[2m", "\033[33m", "\033[31m", "\033[90;9m"


class Agent(nn.Module):
    """Same parameters and names as kvcompression's KVSamplingGroupedQueryAgent (use_kv_queries=False),
    so the trained checkpoints load with strict=True without Apple's repo installed."""

    def __init__(self, head_dim, max_seq_length, mlp_hidden_size, num_mlp_layers, pos_embedding_dim, processing_dim):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.key_projection = nn.Linear(head_dim, processing_dim)
        self.value_projection = nn.Linear(head_dim, processing_dim)
        self.pos_embedding = nn.Embedding(max_seq_length, pos_embedding_dim)
        self.norm_key = nn.LayerNorm(processing_dim)
        self.norm_value = nn.LayerNorm(processing_dim)
        self.norm_pos = nn.LayerNorm(pos_embedding_dim)
        layers, d = [], 2 * processing_dim + pos_embedding_dim
        for _ in range(num_mlp_layers):
            layers += [nn.Linear(d, mlp_hidden_size), nn.LayerNorm(mlp_hidden_size), nn.SiLU()]
            d = mlp_hidden_size
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)


def load_agent(path):
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except ModuleNotFoundError as e:
            raise SystemExit(f"loading {path} needs the module '{e.name}': pip install {e.name}") from e
    cfg = {k: ck["config"]["agent"][k] for k in ("head_dim", "max_seq_length", "mlp_hidden_size",
                                                 "num_mlp_layers", "pos_embedding_dim", "processing_dim",
                                                 "use_kv_queries")}
    if cfg.pop("use_kv_queries") or ck["config"]["agent"].get("use_keydiff_features", False):
        raise ValueError(f"{path}: agent uses queries/keydiff features, which this loader does not support")
    sd = ck["ema_model_state_dict"]
    if sd is None:
        raise ValueError(f"{path}: no EMA weights in checkpoint")
    if any(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items() if k.startswith("module.")}
    agent = Agent(**{k: int(v) for k, v in cfg.items()})
    agent.load_state_dict(sd, strict=True)
    return agent


def load_agents(root, n_layers, n_kv, device):
    root = Path(root)
    if not (root / "layer_000000").is_dir():
        hits = sorted(root.rglob("layer_000000"))
        if not hits:
            raise SystemExit(f"no layer_000000 folder under {root}")
        root = hits[0].parent
    agents = []
    for l in range(n_layers):
        row = []
        for h in range(n_kv):
            cks = sorted((root / f"layer_{l:06d}" / f"kv_head_{h:03d}").glob("**/best_ckpt.pth"))
            if len(cks) != 1:
                raise SystemExit(f"layer {l} head {h}: expected 1 best_ckpt.pth, found {len(cks)}")
            row.append(load_agent(cks[0]).to(device).eval())
        agents.append(row)
    print(f"loaded {n_layers * n_kv} KVP agents from {root}")
    return agents


def make_picker(kind):
    """picker(past_user_texts, new_text) -> index of the earlier message to restore, or None."""
    if kind == "lexical":
        return lambda past, q: choose_turn(past, q)[0]
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise SystemExit("--retriever embed needs: pip install sentence-transformers") from e
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    def pick(past, q):
        e = st.encode(past + [q], normalize_embeddings=True, convert_to_tensor=True)
        return int((e[:-1] @ e[-1]).argmax())

    return pick


class Conversation:
    def __init__(self, name, model, tok, eos_ids, im_end, budget, agents, pool_window, picker=None):
        method, _, suffix = name.partition("+")
        fix = suffix == "fix"
        if suffix and not fix and not (suffix.startswith("pool") and (suffix[4:] == "" or suffix[4:].isdigit())):
            raise SystemExit(f"unknown method suffix '+{suffix}' in '{name}': use +fix, +pool or +poolN, e.g. kvp+pool8")
        if fix and method not in ("kvp", "streaming"):
            raise SystemExit(f"+fix works with kvp or streaming, not {method}")
        pool = (int(suffix[4:]) if suffix[4:] else pool_window) if suffix and not fix else 0
        self.name, self.model, self.tok, self.eos_ids, self.im_end = name, model, tok, eos_ids, im_end
        self.state = EvictingState(model, method, 0 if method == "full" else budget,
                                   agents=agents if method == "kvp" else None, pool_window=pool, offload=fix)
        self.picker = picker
        self.pending, self.stream, self.turn = [], [], 0
        self.user_texts, self.user_ranges, self.note = [], [], ""

    def retrieve(self, text):
        """+fix: restore the best-matching earlier user message into all heads before the new one is read."""
        j = self.picker(self.user_texts, text)
        self.state.force_pos, self.state.n_restored = None, 0
        if j is None:
            return "no earlier message matched; nothing restored"
        R = self.user_ranges[j]
        if len(R) + N_SINK + N_RECENT > self.state.cache_len():
            return f"your message #{j + 1} is too long to restore at this budget; nothing restored"
        held = sum(set(R) <= set(row) for p in self.state.pos for row in p.tolist())
        self.state.restore(R)
        n_heads = len(self.state.pos) * self.state.pos[0].shape[0]
        preview = self.user_texts[j][:60] + ("..." if len(self.user_texts[j]) > 60 else "")
        return f"restored your message #{j + 1} ({len(R)} tokens; it was whole in {held}/{n_heads} heads): {preview!r}"

    def ask(self, text, max_new, show):
        if self.turn == 0:
            prompt = self.tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                                   {"role": "user", "content": text}],
                                                  tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n"
        tt = self.tok(prompt, add_special_tokens=False).input_ids
        self.note = self.retrieve(text) if self.state.offload and self.turn > 0 else ""
        start = self.state.next_pos + len(self.pending)
        if self.turn == 0:
            sys_only = self.tok.apply_chat_template([{"role": "system", "content": SYSTEM}], tokenize=False)
            self.user_ranges.append(list(range(len(self.tok(sys_only, add_special_tokens=False).input_ids), len(tt))))
        else:
            self.user_ranges.append(list(range(start, start + len(tt))))
        self.user_texts.append(text)
        ids = self.pending + tt
        logits = self.state.forward(ids)
        self.stream += ids
        self.state.evict()
        reply, printed = [int(logits.argmax())], ""
        while True:
            text_so_far = self.tok.decode(reply, skip_special_tokens=True)
            if not text_so_far.endswith("�") and text_so_far.startswith(printed):
                if show:
                    print(text_so_far[len(printed):], end="", flush=True)
                printed = text_so_far
            if reply[-1] in self.eos_ids or len(reply) >= max_new:
                break
            logits = self.state.forward([reply[-1]])
            self.stream.append(reply[-1])
            reply.append(int(logits.argmax()))
        final = self.tok.decode(reply, skip_special_tokens=True)
        if show and final.startswith(printed):
            print(final[len(printed):], end="", flush=True)
        printed = final
        self.state.evict()
        self.pending = [self.im_end] if reply[-1] in self.eos_ids else [reply[-1], self.im_end]
        self.turn += 1
        return printed

    def cache_line(self):
        n, seen = self.state.cache_len(), self.state.next_pos
        note = f"; {self.note}" if self.note else ""
        return f"{DIM}[{self.name}] cache {n}/{seen} tokens per head ({n / max(seen, 1):.0%} kept){note}{RESET}"

    def kept_frac(self):
        """Share of (layer, kv-head) pairs that still hold each conversation token, indexed like self.stream."""
        seen = self.state.next_pos
        if len(self.stream) != seen:
            raise RuntimeError(f"stream has {len(self.stream)} tokens but {seen} were fed to the model")
        kept = [0] * seen
        for p in self.state.pos:
            for i in p.flatten().tolist():
                kept[i] += 1
        n_heads = len(self.state.pos) * self.state.pos[0].shape[0]
        return [k / n_heads for k in kept]

    def template_mask(self):
        """Chat-format tokens: <|im_start|>, the role name and its newline, <|im_end|> and its newline."""
        im_start = self.tok.convert_tokens_to_ids("<|im_start|>")
        mask = [False] * len(self.stream)
        for i, t in enumerate(self.stream):
            if t == im_start:
                for j in range(i, min(i + 3, len(mask))):
                    mask[j] = True
            elif t == self.im_end:
                for j in range(i, min(i + 2, len(mask))):
                    mask[j] = True
        return mask

    def memory(self):
        if self.state.next_pos == 0:
            return "(nothing yet)"
        frac, tmpl = self.kept_frac(), self.template_mask()
        out = []
        for t, f in zip(self.stream, frac):
            piece = self.tok.decode([t], skip_special_tokens=False)
            piece = piece.replace("<|im_start|>", "\n> ").replace("<|im_end|>", "⏎")
            color = "" if f >= 0.95 else YELLOW if f >= 0.5 else RED if f > 0 else GREY
            out.append(f"{color}{piece}{RESET}" if color else piece)
        mean = sum(frac) / len(frac)
        n_t = sum(tmpl)
        slots_t = sum(f for f, m in zip(frac, tmpl) if m) / sum(frac)
        legend = (f"{BOLD}legend:{RESET} share of the {len(self.state.pos) * self.state.pos[0].shape[0]} "
                  f"(layer, head) pairs holding the token: normal >= 95%, {YELLOW}yellow{RESET} 50-95%, "
                  f"{RED}red{RESET} 1-50%, {GREY}grey struck-through{RESET} 0%. ⏎ = <|im_end|>\n"
                  f"average token: {mean:.0%} of heads (so red is the typical colour). "
                  f"chat-format tokens: {n_t}/{len(tmpl)} = {n_t / len(tmpl):.0%} of the conversation, "
                  f"{slots_t:.0%} of cache slots")
        return legend + "\n" + "".join(out)

    def find(self, text):
        """Per-token share of heads holding every occurrence of `text` (exact, case-sensitive)."""
        if self.state.next_pos == 0:
            return "(nothing yet)"
        frac = self.kept_frac()
        pieces = [self.tok.decode([t], skip_special_tokens=False) for t in self.stream]
        starts, n = [], 0
        for p in pieces:
            starts.append(n)
            n += len(p)
        full = "".join(pieces)
        held = [[set(row) for row in p.tolist()] for p in self.state.pos]
        lines, at = [], full.find(text)
        while at != -1:
            end = at + len(text)
            toks = [i for i, s in enumerate(starts) if s < end and s + len(pieces[i]) > at]
            per_layer = [sum(all(i in h for i in toks) for h in layer) for layer in held]
            whole = sum(per_layer) / sum(len(layer) for layer in held)
            lines.append(f"at token {toks[0]}: " + "  ".join(f"{pieces[i]!r} {frac[i]:.0%}" for i in toks)
                         + f"  | all {len(toks)} held by {whole:.0%} of heads; heads per layer 0..{len(held) - 1}: "
                         + "".join(str(n) for n in per_layer))
            at = full.find(text, at + 1)
        return "\n".join(lines) if lines else f"{text!r} not found (search is case-sensitive)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="kvp,kvp+fix,full",
                    help="comma list of full, streaming, h2o, tova, kvp; add +fix (kvp/streaming: restore the matching "
                         "earlier message), +pool (window from --pool_window) or +poolN (window N), e.g. kvp,kvp+fix,full")
    ap.add_argument("--retriever", default="embed", choices=["embed", "lexical"], help="for +fix")
    ap.add_argument("--budget", type=int, default=128, help="tokens kept per head")
    ap.add_argument("--agents_dir", default="", help="folder with the 48 trained agents (needed for kvp)")
    ap.add_argument("--pool_window", type=int, default=4)
    ap.add_argument("--max_new", type=int, default=200)
    ap.add_argument("--model", default=MODEL, help="any Hugging Face chat model; KVP agents only fit the model they were trained on")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if os.name == "nt":
        os.system("")
    if transformers.__version__ != "4.55.4":
        raise SystemExit(f"needs transformers==4.55.4 (cache API), found {transformers.__version__}: "
                         "pip install transformers==4.55.4")

    names = [m.strip() for m in args.methods.split(",") if m.strip()]
    bf16_ok = args.device.startswith("cuda") and torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if bf16_ok else torch.float32
    print(f"loading {args.model} on {args.device} ({dtype}) ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, attn_implementation="eager").to(args.device).eval()
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end is None or im_end == tok.unk_token_id:
        raise SystemExit(f"{args.model} does not use the <|im_start|>/<|im_end|> chat format; only Qwen-family models are supported")
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) | {im_end}
    agents = None
    if any(n.startswith("kvp") for n in names):
        if not args.agents_dir:
            raise SystemExit("kvp needs --agents_dir")
        if args.model != MODEL:
            raise SystemExit(f"the KVP agents were trained for {MODEL}; with --model {args.model} use full/streaming/h2o/tova")
        agents = load_agents(args.agents_dir, model.config.num_hidden_layers, model.config.num_key_value_heads,
                             args.device)

    picker = make_picker(args.retriever) if any(n.endswith("+fix") for n in names) else None

    def fresh():
        return [Conversation(n, model, tok, eos_ids, im_end, args.budget, agents, args.pool_window, picker)
                for n in names]

    convs = fresh()
    print(f"\n{BOLD}Chatting with {', '.join(names)} at budget {args.budget} tokens per head.{RESET} "
          f"Tip: tell it a fact, chat about other things, then ask. /memory, /find TEXT, /reset, /quit\n")
    while True:
        try:
            text = input(f"{BOLD}you>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/reset":
            convs = fresh()
            print("new conversation\n")
            continue
        if text.startswith("/memory"):
            parts = text.split()
            pick = [c for c in convs if len(parts) == 1 or c.name == parts[1]]
            for c in pick:
                print(f"\n{BOLD}[{c.name}] memory{RESET}\n{c.memory()}\n")
            continue
        if text.startswith("/find "):
            query = text[len("/find "):]
            for c in convs:
                print(f"{BOLD}[{c.name}]{RESET} {c.find(query)}")
            print()
            continue
        for c in convs:
            print(f"{BOLD}[{c.name}]>{RESET} ", end="", flush=True)
            with torch.no_grad():
                c.ask(text, args.max_new, show=True)
            print()
            print(c.cache_line())
        print()


if __name__ == "__main__":
    main()
