"""Do MT-NIAH filler prompts also appear in the 400 OASST2 dialogues the KVP agents were trained on?

Rebuilds the training set exactly as trace collection did (same dataset class and defaults as
configs/preprocess_oasst2_0_5b.yaml; samples 0-399 = train split) and searches each filler prompt
in the decoded training prompts. Run in the repo env (CPU is fine).
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from torchtune.models.qwen2_5 import qwen2_5_tokenizer

from kvcompression.data.oasst2_dataset import OASST2MultiTurnDataset


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data", required=True, help="mt_niah.jsonl")
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = qwen2_5_tokenizer(path=f"{args.model_dir}/vocab.json", merges_file=f"{args.model_dir}/merges.txt",
                            max_seq_len=None)
    ds = OASST2MultiTurnDataset(tok, n_samples=500, min_prompt_tokens=200, max_prompt_tokens=1200,
                                min_user_turns=2, seed=0)
    train = [tok.decode(ds[i]["tokens"], skip_special_tokens=False) for i in range(args.n_train)]
    exact_blob = "\n\u0000\n".join(train)
    norm_blob = "\n\u0000\n".join(norm(t) for t in train)

    rows = [json.loads(l) for l in open(args.data)]
    fillers = sorted({t for r in rows for t in r["user_turns"][1:-1]})
    exact = {f for f in fillers if f.strip() in exact_blob}
    loose = {f for f in fillers if norm(f) in norm_blob}

    per_k = defaultdict(lambda: [0, 0])
    for r in rows:
        per_k[r["k"]][0] += 1
        per_k[r["k"]][1] += any(t in loose for t in r["user_turns"][1:-1])

    report = {
        "n_unique_fillers": len(fillers),
        "in_training_exact": len(exact),
        "in_training_whitespace_normalized": len(loose),
        "dialogues_with_overlap_by_k": {k: {"dialogues": v[0], "with_overlap": v[1]} for k, v in sorted(per_k.items())},
        "overlapping_fillers": sorted(loose),
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "overlapping_fillers"}, indent=2))


if __name__ == "__main__":
    main()
