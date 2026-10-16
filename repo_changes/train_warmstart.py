"""Apple's RLOO training recipe, started from existing agents instead of random weights.

Goes in src/kvcompression/entrypoints/rl/. Same CLI as train_agent_sampler_distributed.py; the config needs
`init_from`: a folder with layer_XXXXXX/kv_head_XXX/**/best_ckpt.pth (one per agent). The training model starts
from that checkpoint's EMA weights (the ones used at inference); the EMA copy gets the full saved EMA state
(weights + n_averaged), so after burn-in it keeps averaging from the original EMA. Optimizer, scheduler and step
counter start fresh. (Upstream never auto-resumes, so an interrupted agent must be retrained from scratch.)
"""
import sys
from pathlib import Path

import torch

import kvcompression.entrypoints.rl.train_agent_sampler_distributed as base
from kvcompression.utils.utils import get_underlying_model


def find_ckpt(root, layer, head):
    d = Path(root) / f"layer_{layer:06d}" / f"kv_head_{head:03d}"
    hits = sorted(d.glob("**/best_ckpt.pth"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly 1 best_ckpt.pth under {d}, found {len(hits)}")
    return hits[0]


class WarmStartRecipe(base.RLOOTrainingRecipeDistributed):
    def _setup_model_and_agent(self, checkpoint_data):
        super()._setup_model_and_agent(checkpoint_data)
        if checkpoint_data is not None:
            raise RuntimeError("unexpected resume checkpoint in a warm-start run")
        root = self._cfg.get("init_from")
        if not root:
            raise ValueError("train_warmstart.py needs `init_from` in the config")
        path = find_ckpt(root, self._cfg.target_layer_idx, self._cfg.kv_head_idx)
        ck = torch.load(path, map_location=self._device, weights_only=False)
        ema = ck.get("ema_model_state_dict")
        if not ema:
            raise RuntimeError(f"{path}: no ema_model_state_dict")
        sd = {k[len("module."):]: v for k, v in ema.items() if k.startswith("module.")}
        if not sd:
            raise RuntimeError(f"{path}: EMA state has no 'module.' weights")

        underlying = get_underlying_model(self._agent)
        model = underlying._orig_mod if getattr(underlying, "_orig_mod", None) is not None else underlying
        model.load_state_dict(sd, strict=True)
        self._ema_agent.load_state_dict(ema, strict=True)

        for name, ref in sd.items():
            for tgt in (model.state_dict()[name], self._ema_agent.module.state_dict()[name]):
                if not torch.equal(tgt, ref.to(device=tgt.device, dtype=tgt.dtype)):
                    raise RuntimeError(f"warm start did not copy {name}")
        base.log_rank_zero(base.logger, f"Warm start: loaded {len(sd)} tensors from {path}")


base.RLOOTrainingRecipeDistributed = WarmStartRecipe

if __name__ == "__main__":
    sys.exit(base.main())
