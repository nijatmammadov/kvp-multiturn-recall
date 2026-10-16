"""Collect everything needed to evaluate the 48-agent training run into /workspace/kv/export/training_eval.

Copies best_ckpt.pth + figures per agent, configs and logs, and flattens metrics_history
(from each agent's final 4000-step checkpoint) into CSVs. Read-only on the sources.
"""
import csv
import re
import shutil
import tarfile
from pathlib import Path

import torch

KV = Path("/workspace/kv")
REPO = KV / "ml-learning-to-evict"
AG = REPO / "agents/grouped/oasst2_0_5b"
OUT = KV / "export/training_eval"
TRAIN_KEYS = ["policy_loss", "entropy_loss", "total_loss", "episode_reward", "avg_advantage", "episode_length",
              "avg_entropy", "learning_rate", "gradient_norm", "agent_score_mean", "agent_score_std",
              "reward_mean", "reward_std", "reward_min", "reward_max", "rewards_clamping_percentage",
              "adv_clamping_percentage"]

if OUT.exists():
    raise SystemExit(f"{OUT} already exists; not overwriting")
(OUT / "agents").mkdir(parents=True)
(OUT / "metrics").mkdir()
(OUT / "logs").mkdir()
(OUT / "config").mkdir()
(OUT / "probe").mkdir()

final_rows, eval_rows, best_rows = [], [], []
with open(OUT / "metrics/train_curves.csv", "w", newline="") as ft:
    wt = csv.writer(ft)
    wt.writerow(["layer", "head", "step", *TRAIN_KEYS])
    for l in range(24):
        for h in range(2):
            runs = list((AG / f"layer_{l:06d}" / f"kv_head_{h:03d}").iterdir())
            if len(runs) != 1:
                raise RuntimeError(f"L{l} H{h}: {len(runs)} run dirs")
            run = runs[0]
            dst = OUT / "agents" / f"L{l:02d}_H{h}"
            dst.mkdir()
            shutil.copy2(run / "best_ckpt.pth", dst / "best_ckpt.pth")
            shutil.copytree(run / "figures", dst / "figures")
            (dst / "source_run_dir.txt").write_text(str(run) + "\n")

            final_ckpt = sorted((run / "checkpoints").glob("*.pth"))[-1]
            ck = torch.load(final_ckpt, map_location="cpu", weights_only=False)
            m = ck["metrics_history"]
            n = len(m["total_loss"])
            if n != 4000:
                raise RuntimeError(f"L{l} H{h}: {n} train steps in {final_ckpt.name}")
            for i in range(n):
                wt.writerow([l, h, i + 1, *[m[k][i] for k in TRAIN_KEYS]])
            ev = m["eval_reward"]
            for i, r in enumerate(ev):
                eval_rows.append([l, h, 200 * (i + 1), r, m["eval_length"][i]])
            for s, r in zip(m["best_model_step"], m["best_eval_reward"]):
                best_rows.append([l, h, s, r])

            log = (KV / "logs/train" / f"L{l}_H{h}.log").read_text(errors="replace")
            fr = re.search(r"final_reward: (-?[0-9.]+)", log)
            if not fr:
                raise RuntimeError(f"L{l} H{h}: no final_reward in log")
            best = torch.load(run / "best_ckpt.pth", map_location="cpu", weights_only=False)
            final_rows.append([l, h, float(fr.group(1)), best["step"], best["best_eval_reward"], ev[-1],
                               max(ev), m["agent_param_count"][0]])
            print(f"L{l} H{h} done", flush=True)

with open(OUT / "metrics/eval_curves.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["layer", "head", "step", "eval_reward", "eval_length"])
    w.writerows(eval_rows)
with open(OUT / "metrics/best_model_updates.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["layer", "head", "step", "best_eval_reward"])
    w.writerows(best_rows)
with open(OUT / "metrics/final_rewards.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["layer", "head", "final_reward", "best_ckpt_step", "best_eval_reward", "eval_reward_step4000",
                "max_eval_reward", "agent_param_count"])
    w.writerows(final_rows)

with tarfile.open(OUT / "logs/train_logs.tar.gz", "w:gz") as t:
    for p in sorted((KV / "logs/train").glob("L*_H*.log")):
        t.add(p, arcname=f"train/{p.name}")
    for p in sorted((KV / "logs/train").glob("aborted*")):
        t.add(p, arcname=f"train/{p.name}")
for p in ["train_all.sh", "overnight_status.log", "RESULTS.md", "logs/train_all.log", "train_smoke.log"]:
    if (KV / p).exists():
        shutil.copy2(KV / p, OUT / ("logs" if p.endswith(".log") else "config") / Path(p).name)
for p in ["configs/train_0_5b.yaml", "configs/train_single_head.yaml", "configs/preprocess_oasst2_0_5b.yaml",
          "src/kvcompression/data/oasst2_dataset.py"]:
    shutil.copy2(REPO / p, OUT / "config" / Path(p).name)
for p in ["probe.log", "probe_gpu.csv"]:
    shutil.copy2(KV / p, OUT / "probe" / p)
probe = list((REPO / "agents/grouped/probe/layer_000012/kv_head_000").iterdir())[0]
shutil.copy2(probe / "best_ckpt.pth", OUT / "probe" / "best_ckpt_L12_H0.pth")
shutil.copytree(probe / "figures", OUT / "probe" / "figures")

(OUT / "README.md").write_text("""# Training evaluation export: 48 KVP agents (Qwen2.5-0.5B-Instruct, OASST2 multi-turn)

Exported 2026-10-10 from /workspace/kv on the vast.ai instance. Config: configs/train_0_5b.yaml
(4000 steps, P=4 parallel, sweep oasst2_0_5b). Reward: FutureAttentionAucNormalizedReward (closer to 0 is better).

- agents/L{layer}_H{head}/best_ckpt.pth: best-eval checkpoint (model + EMA weights, config, full metrics_history)
- agents/L{layer}_H{head}/figures/: the trainer's training / loss / eval curve plots (SVG)
- metrics/final_rewards.csv: final_reward (best model re-evaluated), best step, last and max eval_reward per agent
- metrics/eval_curves.csv: eval_reward every 200 steps (48 val samples) per agent
- metrics/train_curves.csv: per-step training metrics (4000 steps x 48 agents) from the 4000-step checkpoint
- metrics/best_model_updates.csv: steps where a new best model was saved
- logs/train_logs.tar.gz: raw per-agent training logs; logs/overnight_status.log, train_all.log
- config/: training + preprocessing configs, dataset code, launch script, RESULTS.md (full project log)
- probe/: the single-agent timing probe (L12 H0, run alone on the GPU) with GPU usage CSV
""")
print("export done:", OUT)
