"""CPU tests for batch v6 (head-subset needle protection). Tiny random Qwen2, no downloads.
Run from eval/: uv run --project $P python test_v6.py
"""
import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

import evict_eval as ev
from evict_eval import EvictingState, head_subset

L, KV, BUDGET = 3, 2, 40


class FakeAgent(nn.Module):
    use_kv_queries = False
    use_keydiff_features = False
    max_seq_length = 512

    def __init__(self, head_dim, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.key_projection, self.value_projection = nn.Linear(head_dim, 8), nn.Linear(head_dim, 8)
        self.pos_embedding = nn.Embedding(512, 8)
        self.norm_key, self.norm_value, self.norm_pos = nn.LayerNorm(8), nn.LayerNorm(8), nn.LayerNorm(8)
        self.mlp = nn.Linear(24, 1)


def tiny():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=L,
                      num_attention_heads=4, num_key_value_heads=KV, max_position_embeddings=512)
    cfg._attn_implementation = "eager"
    model = Qwen2ForCausalLM(cfg).eval()
    agents = [[FakeAgent(16, 10 * l + h) for h in range(KV)] for l in range(L)]
    return model, agents


def run(model, agents, method, force=None, heads=None, n_steps=4):
    torch.manual_seed(1)
    st = EvictingState(model, method, BUDGET, agents=agents, protect_needle="value" if force else "",
                       protect_heads=heads)
    st.forward(torch.randint(0, 128, (30,)).tolist())
    if force:
        st.force_pos = torch.tensor(force)
    st.evict()
    for _ in range(n_steps):
        st.forward(torch.randint(0, 128, (25,)).tolist())
        st.evict()
    return st


def test_head_subset():
    a = head_subset("w_k4_001", 24, 2, 12)
    assert a.shape == (24, 2) and a.sum() == 12
    assert torch.equal(a, head_subset("w_k4_001", 24, 2, 12)), "not deterministic"
    for small, big in ((6, 12), (12, 24), (24, 48)):
        assert (head_subset("x", 24, 2, small) <= head_subset("x", 24, 2, big)).all(), "subsets not nested"
    assert not torch.equal(head_subset("a", 24, 2, 6), head_subset("b", 24, 2, 6)), "same subset for every dialogue"
    for bad in (0, 49):
        try:
            head_subset("x", 24, 2, bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_protect_heads():
    model, agents = tiny()
    force = [6, 7, 8, 9, 10]
    mask = head_subset("d0", L, KV, 2)
    st = run(model, agents, "kvp", force, mask)
    for l in range(L):
        for h in range(KV):
            held = all(p in st.pos[l][h].tolist() for p in force)
            if mask[l, h]:
                assert held, f"protected head ({l}, {h}) lost a forced position"
    all_heads = run(model, agents, "kvp", force, torch.ones(L, KV, dtype=torch.bool))
    no_mask = run(model, agents, "kvp", force, None)
    assert all(torch.equal(a, b) for a, b in zip(all_heads.pos, no_mask.pos)), "all-heads mask != plain protect_needle"
    plain = run(model, agents, "kvp")
    lost = [(l, h) for l in range(L) for h in range(KV)
            if not all(p in plain.pos[l][h].tolist() for p in force)]
    assert lost, "test is vacuous: plain kvp keeps the forced positions everywhere; change the setup"


def test_protect_heads_needs_protect_needle():
    model, agents = tiny()
    try:
        EvictingState(model, "kvp", BUDGET, agents=agents, protect_heads=torch.ones(L, KV, dtype=torch.bool))
    except ValueError:
        return
    raise AssertionError("protect_heads without protect_needle was accepted")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
