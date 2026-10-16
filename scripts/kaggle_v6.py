"""Batch v6 breadth ladder on Kaggle (fp32 on T4/P100).

Runs, all on the word set q1, b = 256, k in {4, 8}, fp32, fine-tuned agents:
  full cache, kvp_ft (n = 0), whole needle turn force-kept in n = 48, 24, 12, 6 heads, name + city in 48 heads.
Uses every visible GPU (one config per GPU at a time). After each config: commit + push its JSONL and log
(unless --no_push). evict_eval.py resumes by dialogue id, so a rerun after a session ends continues where it stopped.

usage (repo root): python scripts/kaggle_v6.py --apple /kaggle/working/ml-learning-to-evict [--no_push]
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = "results/mt_niah_words_lead2_q1.jsonl"
AGENTS = "agents/oasst2_recall_ft_0_5b"
OUT = "results/logs_eval_v6k_fp32"
COMMON = ["--data", DATA, "--out_dir", OUT, "--dtype", "fp32", "--ks", "4,8"]
KVP = ["--method", "kvp", "--budget", "256", "--agents_dir", AGENTS]
CONFIGS = [
    ("full_b0", ["--method", "full", "--budget", "0"]),
    ("kvp_ft_turn_heads48_b256", KVP + ["--protect_needle", "turn", "--protect_heads", "48", "--tag", "ft_turn_heads48"]),
    ("kvp_ft_b256", KVP + ["--tag", "ft"]),
    ("kvp_ft_turn_heads24_b256", KVP + ["--protect_needle", "turn", "--protect_heads", "24", "--tag", "ft_turn_heads24"]),
    ("kvp_ft_turn_heads12_b256", KVP + ["--protect_needle", "turn", "--protect_heads", "12", "--tag", "ft_turn_heads12"]),
    ("kvp_ft_turn_heads6_b256", KVP + ["--protect_needle", "turn", "--protect_heads", "6", "--tag", "ft_turn_heads6"]),
    ("kvp_ft_heads48_b256", KVP + ["--protect_needle", "value", "--protect_heads", "48", "--tag", "ft_heads48"]),
]
N_DIALOGUES = 120


def sh(cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True, **kw)


def env_for(apple, gpu=None):
    env = dict(os.environ, KVP_AGENT_LOADER="light",
               PYTHONPATH=os.pathsep.join([str(Path(apple) / "src"), str(REPO / "eval"), os.environ.get("PYTHONPATH", "")]))
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def check_env(apple):
    import torch
    import transformers

    if transformers.__version__ != "4.55.4":
        sys.exit(f"transformers {transformers.__version__} != 4.55.4 (pip install transformers==4.55.4, then restart)")
    if not torch.cuda.is_available():
        sys.exit("no CUDA GPU: set Accelerator to GPU in the notebook settings")
    if not (Path(apple) / "src/kvcompression/rl/agents/kv_sorting_agent.py").is_file():
        sys.exit(f"Apple repo not found at {apple}")
    n = sum(1 for _ in (REPO / AGENTS).glob("layer_*/kv_head_*/best_ckpt.pth"))
    if n != 48:
        sys.exit(f"need 48 agents in {AGENTS}, found {n}")
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(f"torch {torch.__version__}, transformers {transformers.__version__}, GPUs {gpus}", flush=True)
    return len(gpus)


def run_config(name, args, apple, gpu, log_dir, out=OUT, extra=()):
    log = log_dir / f"{name}.log"
    cmd = [sys.executable, "eval/evict_eval.py"] + COMMON + list(args) + list(extra)
    if out != OUT:
        cmd[cmd.index("--out_dir") + 1] = out
    t0 = time.time()
    with log.open("a") as f:
        f.write(f"$ {' '.join(cmd)}  (GPU {gpu})\n")
        f.flush()
        r = subprocess.run(cmd, cwd=REPO, env=env_for(apple, gpu), stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"{name} failed (exit {r.returncode}); see {log}")
    return time.time() - t0


def push(paths, msg, enabled, lock):
    if not enabled:
        return
    with lock:
        sh(["git", "add", "-f", *[str(p) for p in paths]])
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode == 0:
            return
        sh(["git", "commit", "-q", "-m", msg])
        for attempt in range(3):
            try:
                sh(["git", "pull", "-q", "--rebase", "origin", "main"])
                sh(["git", "push", "-q", "origin", "HEAD:main"])
                return
            except subprocess.CalledProcessError:
                if attempt == 2:
                    raise
                time.sleep(10)


def summary(name):
    p = REPO / OUT / f"{name}_fp32.jsonl"
    recs = [json.loads(l) for l in p.open()]
    return f"{sum(r['correct'] for r in recs)}/{len(recs)} correct (unconditional), {len(recs)} dialogues"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apple", required=True, help="clone of apple/ml-learning-to-evict at d33c2dc")
    ap.add_argument("--no_push", action="store_true")
    ap.add_argument("--skip_speed", action="store_true")
    args = ap.parse_args()
    n_gpu = check_env(args.apple)
    log_dir = REPO / "logs" / "kaggle_v6"
    log_dir.mkdir(parents=True, exist_ok=True)
    (REPO / OUT).mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    sh([sys.executable, "eval/test_v6.py"], env=env_for(args.apple))

    if not args.skip_speed:
        speed_out = "logs_speed_v6k"
        per = {}
        for name, a in (CONFIGS[0], CONFIGS[2]):
            sec = run_config(f"speed_{name}", a, args.apple, 0, log_dir, out=speed_out, extra=["--limit", "1"])
            per[name] = sec / 2
        kvp_sec = per["kvp_ft_b256"]
        total = (per["full_b0"] + (len(CONFIGS) - 1) * kvp_sec) * N_DIALOGUES / n_gpu
        print(f"speed (incl. model/agent loading): full {per['full_b0']:.0f} s/dialogue, kvp {kvp_sec:.0f} s/dialogue; "
              f"estimated total {total / 3600:.1f} h on {n_gpu} GPU(s). Kaggle sessions stop after 12 h; "
              "rerun this script in a new session to resume.", flush=True)

    jobs = queue.Queue()
    for c in CONFIGS:
        jobs.put(c)
    errors = []

    def worker(gpu):
        while not errors:
            try:
                name, a = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                sec = run_config(name, a, args.apple, gpu, log_dir)
                line = f"{name}: {summary(name)}, {sec / 60:.0f} min on GPU {gpu}"
                print(line, flush=True)
                push([Path(OUT) / f"{name}_fp32.jsonl", log_dir / f"{name}.log"], f"v6k (Kaggle fp32): {line}",
                     not args.no_push, lock)
            except Exception as e:
                errors.append(e)
                print(f"ERROR in {name}: {e}", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in range(n_gpu)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]

    out = []
    for cmd in ([sys.executable, "eval/head_breadth.py", "--base", OUT, "--breadth", OUT, "--ns", "6", "12", "24", "48",
                 "--prefix", "kvp_ft_turn_heads", "--dtype", "fp32", "--protected", "whole needle turn",
                 "--out", "results/head_breadth_v6k_fp32.md"],
                [sys.executable, "eval/paired_stats.py", "--logs", OUT, "--condition_on", "full:0",
                 "--pairs", "kvp_ft_turn_heads48:kvp_ft_heads48", "kvp_ft_turn_heads48:kvp_ft",
                 "--pool_ks", "4,8", "--out", "results/paired_stats_v6k_binding.md"]):
        sh(cmd)
    new = {json.loads(l)["id"]: json.loads(l) for l in (REPO / OUT / "kvp_ft_b256_fp32.jsonl").open()}
    old = {json.loads(l)["id"]: json.loads(l) for l in (REPO / "results/logs_eval_v5_words_q1/kvp_ft_b256_bf16.jsonl").open()}
    same = sum(new[i]["answer"] == old[i]["answer"] for i in new)
    note = REPO / "results/v6k_fp32_vs_bf16.md"
    note.write_text(f"kvp_ft b=256 words q1 k in {{4, 8}}: fp32 (Kaggle) vs bf16 (v5, RTX 4090): {same}/{len(new)} "
                    "identical answers. Descriptive only; the fp32 ladder is analysed on its own.\n")
    push(["results/head_breadth_v6k_fp32.md", "results/paired_stats_v6k_binding.md", note.relative_to(REPO)],
         "v6k (Kaggle fp32): breadth ladder analysis", not args.no_push, lock)

    tar = Path("/kaggle/working/results_v6k_npz.tar.gz") if Path("/kaggle/working").is_dir() else REPO / "results_v6k_npz.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(REPO / OUT, arcname=OUT)
    print(f"done. .npz archive: {tar} ({tar.stat().st_size / 2**20:.0f} MB) - download it from the notebook output.")
    print((REPO / "results/head_breadth_v6k_fp32.md").read_text())


if __name__ == "__main__":
    main()
