"""CPU tests for batch v7 (--offload_retrieve). Tiny random Qwen2, no downloads. Run from eval/: python test_v7.py"""
import torch

from evict_eval import EvictingState
from offload import choose_turn
from test_v6 import BUDGET, KV, L, tiny


def test_choose_turn():
    past = ["Tell me a joke about cats in the city.", "Please remember this for later: the access code for the silent "
            "harbor C6 is 32162.", "How do I bake bread at home?"]
    assert choose_turn(past, "What is the access code for the silent harbor C6? Answer with the number only.")[0] == 1
    past = ["Which city has the best food?", "My cousin's hamster is called Lady Waffle and lives in Bergen.",
            "Explain how a city council works."]
    assert choose_turn(past, "What is my cousin's hamster called, and in which city does it live?")[0] == 1
    assert choose_turn(past, "What is the weather like?") == (None, 0.0), "one or no shared word must not retrieve"
    assert choose_turn(["code code", "code harbor"], "code harbor")[0] == 1


def offload_state(model, agents):
    torch.manual_seed(1)
    st = EvictingState(model, "kvp", BUDGET, agents=agents, offload=True)
    st.forward(torch.randint(0, 128, (30,)).tolist())
    return st


def test_archive_matches_cache_before_eviction():
    model, agents = tiny()
    st = offload_state(model, agents)
    for l in range(L):
        assert torch.equal(st.arch_k[l], st.cache.layers[l].keys[0].cpu())
        assert torch.equal(st.arch_v[l], st.cache.layers[l].values[0].cpu())


def test_restore():
    model, agents = tiny()
    st = offload_state(model, agents)
    st.evict()
    for _ in range(4):
        st.forward(torch.randint(0, 128, (25,)).tolist())
        st.evict()
    R = list(range(6, 14))
    lost = [(l, h) for l in range(L) for h in range(KV) if not all(p in st.pos[l][h].tolist() for p in R)]
    assert lost, "test is vacuous: nothing to restore"
    before = {(l, h): {int(p): (st.cache.layers[l].keys[0, h, i].clone(), st.cache.layers[l].values[0, h, i].clone())
                       for i, p in enumerate(st.pos[l][h])} for l in range(L) for h in range(KV)}
    n = st.cache_len()
    st.restore(R)
    assert st.cache_len() == n and st.n_restored == len(R)
    for l in range(L):
        layer = st.cache.layers[l]
        assert layer.keys.shape[2] == n and layer.values.shape[2] == n
        for h in range(KV):
            p = st.pos[l][h].tolist()
            assert p == sorted(set(p)), "positions must stay sorted and unique"
            assert all(r in p for r in R), f"head ({l}, {h}) misses a restored position"
            for i, q in enumerate(p):
                k, v = layer.keys[0, h, i], layer.values[0, h, i]
                if q in before[(l, h)]:
                    assert torch.equal(k, before[(l, h)][q][0]) and torch.equal(v, before[(l, h)][q][1])
                else:
                    assert torch.equal(k, st.arch_k[l][h, q]) and torch.equal(v, st.arch_v[l][h, q])
            displaced = set(before[(l, h)]) - set(p)
            assert len(displaced) == sum(r not in before[(l, h)] for r in R)
    st.forward(torch.randint(0, 128, (25,)).tolist())
    st.evict()
    for l in range(L):
        for h in range(KV):
            assert all(r in st.pos[l][h].tolist() for r in R), "restored positions must survive the next eviction"


def test_restore_noop_when_present():
    model, agents = tiny()
    st = offload_state(model, agents)
    pos_before = [p.clone() for p in st.pos]
    keys_before = [st.cache.layers[l].keys.clone() for l in range(L)]
    st.restore([3, 4, 5])
    assert all(torch.equal(a, b) for a, b in zip(pos_before, st.pos))
    assert all(torch.equal(keys_before[l], st.cache.layers[l].keys) for l in range(L))


def test_restore_streaming():
    model, _ = tiny()
    torch.manual_seed(1)
    st = EvictingState(model, "streaming", BUDGET, offload=True)
    st.forward(torch.randint(0, 128, (30,)).tolist())
    for _ in range(4):
        st.evict()
        st.forward(torch.randint(0, 128, (25,)).tolist())
    st.evict()
    R = list(range(6, 14))
    assert not any(r in st.pos[0][0].tolist() for r in R), "streaming should have evicted old positions"
    n = st.cache_len()
    st.restore(R)
    for l in range(L):
        for h in range(KV):
            p = st.pos[l][h].tolist()
            assert len(p) == n and all(r in p for r in R) and p == sorted(set(p))
            for i, q in enumerate(p):
                if q in R:
                    assert torch.equal(st.cache.layers[l].keys[0, h, i], st.arch_k[l][h, q])
    st.forward(torch.randint(0, 128, (25,)).tolist())
    st.evict()
    assert all(r in st.pos[l][h].tolist() for l in range(L) for h in range(KV) for r in R)


def test_offload_needs_kvp():
    model, agents = tiny()
    for kw in ({"method": "h2o"}, {"method": "kvp", "protect_needle": "value"}, {"method": "kvp", "pool_window": 4}):
        m = kw.pop("method")
        try:
            EvictingState(model, m, BUDGET, agents=agents, offload=True, **kw)
        except ValueError:
            continue
        raise AssertionError(f"offload accepted with {m} {kw}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
