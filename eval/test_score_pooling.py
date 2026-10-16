"""CPU tests for score pooling. Run: python test_score_pooling.py"""
import torch

from score_pooling import pool_scores

RAW = [0.40, 0.50, 0.45, 0.60, 0.55, 0.50, 0.42, 0.48, 0.52, 0.58, 0.10, 0.20, 0.95, 0.15, 0.05]


def top(keys, b):
    return sorted(keys.topk(b, dim=-1).indices[0].tolist())


def test_worked_example():
    s = torch.tensor([RAW])
    pos = torch.arange(15)[None]
    assert top(s, 7) == [1, 3, 4, 5, 8, 9, 12], top(s, 7)
    assert top(pool_scores(s, pos, 2), 7) == [3, 4, 10, 11, 12, 13, 14], top(pool_scores(s, pos, 2), 7)


def test_window_zero_keeps_raw_order():
    s = torch.tensor([RAW])
    pos = torch.arange(15)[None]
    assert top(pool_scores(s, pos, 0), 7) == top(s, 7)


def test_gaps_in_positions_block_pooling():
    s = torch.tensor([[0.1, 0.9, 0.2]])
    pos = torch.tensor([[0, 10, 20]])
    keys = pool_scores(s, pos, 4)
    assert top(keys, 1) == [1]
    assert top(keys, 2) == [1, 2], "the 0.9 token is 10 positions away and must not lift its neighbours"


def test_per_head_independent():
    s = torch.tensor([[0.9, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.9]])
    pos = torch.arange(4).expand(2, 4).contiguous()
    keys = pool_scores(s, pos, 1)
    assert keys[0].topk(2).indices.sort().values.tolist() == [0, 1]
    assert keys[1].topk(2).indices.sort().values.tolist() == [2, 3]


def test_rejects_bad_input():
    for bad in (lambda: pool_scores(torch.tensor([[0.1, float("nan")]]), torch.tensor([[0, 1]]), 1),
                lambda: pool_scores(torch.tensor([[0.1, 0.2]]), torch.tensor([[1, 1]]), 1),
                lambda: pool_scores(torch.tensor([[0.1, 0.2]]), torch.tensor([[0, 1]]), -1)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
