"""Concatenation of several QKVDataset sources, each with its own prompt-length rule.

Goes in src/kvcompression/data/. Used to fine-tune the OASST2 agents on recall traces while replaying OASST2
(randomized cut points, as in the original run) so the agents keep what they learned about chat. Each source is a
dict of QKVDataset arguments; layer/head/dtype/mmap are shared.
"""
import bisect

import torch
from torch.utils.data import Dataset

from kvcompression.data.activations_dataset import QKVDataset


class MixedQKVDataset(Dataset):
    def __init__(self, sources, dtype, target_layer_idx, kv_head_idx, query_idx_within_group=None,
                 keep_dim=False, mmap=True, **kwargs):
        if not sources:
            raise ValueError("MixedQKVDataset needs at least one source")
        self.parts = [
            QKVDataset(dtype=dtype, target_layer_idx=target_layer_idx, kv_head_idx=kv_head_idx,
                       query_idx_within_group=query_idx_within_group, keep_dim=keep_dim, mmap=mmap, **dict(src))
            for src in sources
        ]
        self.offsets = []
        n = 0
        for part in self.parts:
            self.offsets.append(n)
            n += len(part)
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if not 0 <= idx < self.n:
            raise IndexError(idx)
        j = bisect.bisect_right(self.offsets, idx) - 1
        sample = self.parts[j][idx - self.offsets[j]]
        sample["id"] = torch.tensor(idx)
        return sample
