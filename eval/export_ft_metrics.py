"""Export the metrics of a warm-start fine-tune (48 agents) into CSVs in the repo, like training/.

  uv run --project $P python eval/export_ft_metrics.py --agents_dir $P/agents/grouped/oasst2_recall_ft_0_5b \
      --logs_dir /workspace/kv/logs/train_0_5b_ft --steps 1500 --eval_interval 100 --out training_ft

Also reports how many agents kept their best checkpoint at the first eval (step == eval_interval). The EMA is not
updated before step 100 (ema_burnin_steps), so such agents are bit-identical to the original agents.
"""
import argparse
import csv
import gzip
import re
from pathlib import Path

import torch

TRAIN_KEYS = ["policy_loss", "total_loss", "episode_reward", "avg_entropy", "learning_rate", "gradient_norm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents_dir", required=True)
    ap.add_argument("--logs_dir", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--eval_interval", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting")
    out.mkdir(parents=True)

    eval_rows, best_rows, final_rows = [], [], []
    with gzip.open(out / "train_curves.csv.gz", "wt", newline="") as fz:
        wt = csv.writer(fz)
        wt.writerow(["layer", "head", "step", *TRAIN_KEYS])
        for l in range(24):
            for h in range(2):
                runs = [p for p in (Path(args.agents_dir) / f"layer_{l:06d}" / f"kv_head_{h:03d}").iterdir()
                        if p.is_dir()]
                if len(runs) != 1:
                    raise SystemExit(f"L{l} H{h}: {len(runs)} run dirs")
                run = runs[0]
                final_ckpt = run / "checkpoints" / f"{args.steps:012d}.pth"
                if not final_ckpt.exists():
                    raise SystemExit(f"L{l} H{h}: missing {final_ckpt} (run did not finish)")
                m = torch.load(final_ckpt, map_location="cpu", weights_only=False)["metrics_history"]
                n = len(m["total_loss"])
                if n != args.steps:
                    raise SystemExit(f"L{l} H{h}: {n} train steps, expected {args.steps}")
                for i in range(n):
                    wt.writerow([l, h, i + 1, *[m[k][i] if k in m else "" for k in TRAIN_KEYS]])
                ev = m["eval_reward"]
                for i, r in enumerate(ev):
                    eval_rows.append([l, h, args.eval_interval * (i + 1), r])
                for s, r in zip(m["best_model_step"], m["best_eval_reward"]):
                    best_rows.append([l, h, s, r])
                best = torch.load(run / "best_ckpt.pth", map_location="cpu", weights_only=False)
                log = (Path(args.logs_dir) / f"L{l}_H{h}.log").read_text(errors="replace")
                fr = re.search(r"final_reward:\s+(?:\S+\.py:\d+\s+)?(-?\d+\.\d+)", log)  # also matches rich-wrapped log lines
                if not fr:
                    raise SystemExit(f"L{l} H{h}: no final_reward in log")
                final_rows.append([l, h, float(fr.group(1)), best["step"], best["best_eval_reward"], ev[0], ev[-1]])

    for name, header, rows in [
        ("eval_curves.csv", ["layer", "head", "step", "eval_reward"], eval_rows),
        ("best_model_updates.csv", ["layer", "head", "step", "best_eval_reward"], best_rows),
        ("final_rewards.csv", ["layer", "head", "final_reward", "best_ckpt_step", "best_eval_reward",
                               "eval_reward_first", "eval_reward_last"], final_rows),
    ]:
        with open(out / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    unchanged = sum(r[3] == args.eval_interval for r in final_rows)
    print(f"exported {len(final_rows)} agents to {out}")
    print(f"agents whose best checkpoint is the first eval (= original weights): {unchanged}/{len(final_rows)}")
    print(f"mean eval reward first -> last: {sum(r[5] for r in final_rows) / 48:.4f} -> "
          f"{sum(r[6] for r in final_rows) / 48:.4f}")


if __name__ == "__main__":
    main()
