import json
import sys
from types import SimpleNamespace

import torch
from transformers import DynamicCache

import evict_eval as ev
from evict_eval import EvictingState, load_model, run_dialogue

DATA = sys.argv[1] if len(sys.argv) > 1 else "/workspace/kv/mt_niah.jsonl"


def test_gather_cpu():
    """Per-head gather keeps exactly the top-b rows per head, protected rows included, order preserved."""
    torch.manual_seed(0)
    model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=2, num_key_value_heads=2,
                                                   num_attention_heads=14), device=torch.device("cpu"))
    st = EvictingState(model, "random_test", budget=30)
    n, d = 100, 8
    st.cache = DynamicCache()
    scores = {}
    for l in range(2):
        pos = torch.arange(n).expand(2, -1).clone()
        k = pos[None, :, :, None].float().expand(1, 2, n, d).clone()
        st.cache.update(k, k + 1000, l)
        st.pos[l] = pos
        scores[l] = torch.rand(2, n)
    ev.SCORERS["random_test"] = lambda s, l: scores[l]
    st.evict()
    for l in range(2):
        layer = st.cache.layers[l]
        for h in range(2):
            s = scores[l][h].clone()
            s[:ev.N_SINK] = float("inf")
            s[n - ev.N_RECENT:] = float("inf")
            expected = s.topk(30).indices.sort().values
            assert torch.equal(st.pos[l][h], expected), f"layer {l} head {h}: positions"
            assert torch.equal(layer.keys[0, h, :, 0].long(), expected), f"layer {l} head {h}: keys"
            assert torch.equal(layer.values[0, h, :, 0].long() - 1000, expected), f"layer {l} head {h}: values"
        assert not torch.equal(st.pos[l][0], st.pos[l][1]), "heads should differ with random scores"
    print("D gather (CPU): PASS")


def replay(model, stream, allowed=None):
    n = len(stream)
    ids = torch.tensor([stream], device=model.device)
    pos = torch.arange(n, device=model.device)[None]
    if allowed is None:
        return model(input_ids=ids, position_ids=pos, use_cache=False).logits[0]
    mask = torch.zeros(n, n, dtype=model.dtype, device=model.device)
    mask.masked_fill_(~allowed, torch.finfo(model.dtype).min)
    return model(input_ids=ids, position_ids=pos, attention_mask=mask[None, None], use_cache=False).logits[0]


def allowed_from_log(visible_log, n):
    allowed = torch.zeros(n, n, dtype=torch.bool)
    for before, new in visible_log:
        for q in new:
            if before is not None:
                allowed[q, before] = True
            allowed[q, [p for p in new if p <= q]] = True
    return allowed


def compare(trace, ref):
    worst, flips = 0.0, 0
    for q, row in trace:
        r = ref[q].float().cpu()
        worst = max(worst, (row - r).abs().max().item())
        flips += int(row.argmax() != r.argmax())
    return worst, flips


def main():
    tok, model, eos_ids, im_end = load_model(torch.float32)
    rows = [json.loads(l) for l in open(DATA) if json.loads(l)["k"] == 8]
    results = []

    test_gather_cpu()

    row = rows[0]
    st = EvictingState(model, "full", 0)
    trace = []
    res = run_dialogue(st, tok, row, eos_ids, im_end, trace=trace)
    with torch.no_grad():
        ref = replay(model, res["stream"])
        n = len(res["stream"])
        ref4d = replay(model, res["stream"], torch.tril(torch.ones(n, n, dtype=torch.bool, device=model.device)))
    worst_a, flips_a = compare(trace, ref)
    b_diff = (ref - ref4d).abs().max().item()
    print(f"A full-cache incremental vs one-shot forward: max logit diff {worst_a:.2e}, argmax flips {flips_a}")
    print(f"B explicit 4D causal mask vs default mask: max logit diff {b_diff:.2e}")
    results += [("A", worst_a < 1e-3 and flips_a == 0), ("B", b_diff < 1e-4)]

    for row in rows[:3]:
        st = EvictingState(model, "streaming", 128, record=True)
        trace = []
        res = run_dialogue(st, tok, row, eos_ids, im_end, trace=trace)
        n = len(res["stream"])
        if st.next_pos != n:
            raise RuntimeError(f"stream length {n} != positions used {st.next_pos}")
        allowed = allowed_from_log(st.visible_log, n).to(model.device)
        with torch.no_grad():
            ref = replay(model, res["stream"], allowed)
        worst_c, flips_c = compare(trace, ref)
        evicted = int((~allowed).tril().sum().item())
        print(f"C {row['id']}: streaming b=128, len {n}, masked (q,k) pairs {evicted}, "
              f"max logit diff {worst_c:.2e}, argmax flips {flips_c}")
        results.append((f"C {row['id']}", evicted > 0 and worst_c < 1e-2 and flips_c == 0))

    failed = [name for name, ok in results if not ok]
    if failed:
        raise SystemExit(f"FAILED: {failed}")
    print("ALL PASS")


if __name__ == "__main__":
    main()
