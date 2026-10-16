import torch


def pool_scores(scores: torch.Tensor, pos: torch.Tensor, window: int) -> torch.Tensor:
    """Max-pool per-token scores over neighbours within `window` ORIGINAL positions.

    scores: [H, n] float, higher = keep.
    pos:    [H, n] int, original positions, strictly increasing along dim -1 per head.
    Returns [H, n] float ranking keys (higher = keep): primary key = pooled score,
    tie-break = raw score, then earlier position.
    """
    if window < 0:
        raise ValueError(f"window must be >= 0, got {window}")
    if scores.shape != pos.shape or scores.dim() != 2:
        raise ValueError(f"scores {tuple(scores.shape)} and pos {tuple(pos.shape)} must both be [H, n]")
    if not torch.isfinite(scores).all():
        raise ValueError("scores contain NaN or inf")
    if (pos[:, 1:] <= pos[:, :-1]).any():
        raise ValueError("pos must be strictly increasing per head")

    H, n = scores.shape
    pooled = scores.clone()
    for d in range(1, window + 1):
        if d >= n:
            break
        close = (pos[:, d:] - pos[:, :-d]) <= window
        cand = torch.where(close, scores[:, d:], torch.full_like(scores[:, d:], float("-inf")))
        pooled[:, :-d] = torch.maximum(pooled[:, :-d], cand)
        cand = torch.where(close, scores[:, :-d], torch.full_like(scores[:, :-d], float("-inf")))
        pooled[:, d:] = torch.maximum(pooled[:, d:], cand)

    order = torch.argsort(pos, dim=-1, descending=False, stable=True)
    order = order.gather(-1, torch.argsort(scores.gather(-1, order), dim=-1, descending=True, stable=True))
    order = order.gather(-1, torch.argsort(pooled.gather(-1, order), dim=-1, descending=True, stable=True))
    keys = torch.empty_like(scores)
    keys.scatter_(-1, order, -torch.arange(n, dtype=scores.dtype, device=scores.device).expand(H, n).contiguous())
    return keys
