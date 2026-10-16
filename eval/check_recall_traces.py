"""Check the recall traces and write their train/val/test split files.

Run from Apple's repo root inside its env:
  uv run python /workspace/kv/kvp-eviction/eval/check_recall_traces.py \
      --config configs/preprocess_recall_0_5b.yaml \
      --trace_dir data/gqa_safetensors/recall_niah/Qwen2.5-0.5B-Instruct/temperature_0.00

Rebuilds the (deterministic) dataset to get the expected answers, checks every stored prompt length matches, reports
how often the model's greedy continuation contains all expected values (per variant), and writes
train.txt (first 500), val.txt (next 50), test.txt (last 50). Exits non-zero if overall accuracy < 0.5: the answer
tokens would then rarely attend to the needle, so the traces would not teach the agents to keep it.
"""
import argparse
import collections
from pathlib import Path

import safetensors.torch
from omegaconf import OmegaConf
from torchtune.models.qwen2_5 import qwen2_5_tokenizer

from kvcompression.data.recall_dataset import RecallNIAHDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--trace_dir", required=True)
    ap.add_argument("--n_train", type=int, default=500)
    ap.add_argument("--n_val", type=int, default=50)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    model_dir = Path("models") / cfg.model_name
    tok = qwen2_5_tokenizer(path=str(model_dir / "vocab.json"), merges_file=str(model_dir / "merges.txt"),
                            max_seq_len=None)
    params = {k: v for k, v in OmegaConf.to_container(cfg.dataset).items() if k != "_component_"}
    ds = RecallNIAHDataset(tok, **params)

    trace_dir = Path(args.trace_dir)
    ok = collections.Counter()
    tot = collections.Counter()
    for i in range(len(ds)):
        d = trace_dir / f"sample_{i:07d}"
        n_prompt = int(safetensors.torch.load_file(d / "prompt_ntokens.safetensors")["tensor"])
        if n_prompt != len(ds.samples[i][0]):
            raise SystemExit(f"{d}: stored prompt has {n_prompt} tokens, rebuilt dataset has {len(ds.samples[i][0])}")
        gen = (d / "generated_text.txt").read_text()
        hit = all(a in gen for a in ds.answers[i])
        for key in (ds.variants[i], "all"):
            tot[key] += 1
            ok[key] += hit
    for key in sorted(tot):
        print(f"{key:>10}: {ok[key]}/{tot[key]} = {ok[key] / tot[key]:.3f}")

    n = len(ds)
    if args.n_train + args.n_val >= n:
        raise SystemExit(f"n_train + n_val must leave a test split ({n} samples)")
    splits = {"train.txt": range(0, args.n_train), "val.txt": range(args.n_train, args.n_train + args.n_val),
              "test.txt": range(args.n_train + args.n_val, n)}
    for name, idx in splits.items():
        text = "".join(f"sample_{i:07d}\n" for i in idx)
        path = trace_dir / name
        if path.exists() and path.read_text() != text:
            raise SystemExit(f"{path} exists with different content")
        path.write_text(text)
    print("splits:", {k: len(v) for k, v in splits.items()})

    if ok["all"] / tot["all"] < 0.5:
        raise SystemExit(f"overall accuracy {ok['all'] / tot['all']:.3f} < 0.5: recall traces carry a weak signal")


if __name__ == "__main__":
    main()
