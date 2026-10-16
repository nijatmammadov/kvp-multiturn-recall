"""Batch v8 on Kaggle (fp32): the mechanism on Qwen2.5-1.5B-Instruct vs 0.5B, StreamingLLM base. No KVP agents.

12 configs (words q1, b = 256, k in {4, 8}): per model full, streaming, streaming + value / keyval / turn
(protected in all heads), streaming + offload. 1.5B first. Pushes after each config; rerun resumes.

usage (repo root): python scripts/kaggle_v8.py [--no_push]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_v6 import REPO, push, sh  # noqa: E402

WORDS = "results/mt_niah_words_lead2_q1.jsonl"
MODELS = {"1p5b": "Qwen/Qwen2.5-1.5B-Instruct", "0p5b": "Qwen/Qwen2.5-0.5B-Instruct"}
ST = ["--method", "streaming", "--budget", "256"]
VARIANTS = [
    ("full_b0", ["--method", "full", "--budget", "0"]),
    ("streaming_value_b256", ST + ["--protect_needle", "value", "--tag", "value"]),
    ("streaming_turn_b256", ST + ["--protect_needle", "turn", "--tag", "turn"]),
    ("streaming_keyval_b256", ST + ["--protect_needle", "keyvalue", "--tag", "keyval"]),
    ("streaming_b256", ST),
    ("streaming_offload_b256", ST + ["--offload_retrieve", "--tag", "offload"]),
]
CONFIGS = [(f"results/logs_eval_v8_{m}_fp32", MODELS[m], name, a) for m in MODELS for name, a in VARIANTS]


def run(out, model, name, a, gpu, log_dir, extra=()):
    log = log_dir / f"{Path(out).name}__{name}.log"
    cmd = [sys.executable, "eval/evict_eval.py", "--data", WORDS, "--out_dir", out, "--dtype", "fp32", "--ks", "4,8",
           "--model", model, *a, *extra]
    t0 = time.time()
    with log.open("a") as f:
        f.write(f"$ {' '.join(cmd)}  (GPU {gpu})\n")
        f.flush()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO / "eval"))
        r = subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"{out}/{name} failed (exit {r.returncode}); see {log}")
    return time.time() - t0, log


def check_env():
    import torch
    import transformers
    from transformers import AutoTokenizer

    if transformers.__version__ != "4.55.4":
        sys.exit(f"transformers {transformers.__version__} != 4.55.4")
    if not torch.cuda.is_available():
        sys.exit("no CUDA GPU")
    rows = [json.loads(l) for l in (REPO / WORDS).open()]
    a, b = AutoTokenizer.from_pretrained(MODELS["0p5b"]), AutoTokenizer.from_pretrained(MODELS["1p5b"])
    texts = [t for r in rows for t in r["user_turns"]] + [rows[0]["system"]]
    if any(a(t).input_ids != b(t).input_ids for t in texts):
        sys.exit("the 1.5B tokenizer differs from 0.5B: span checks were validated for 0.5B only; stop and report")
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(f"torch {torch.__version__}, transformers {transformers.__version__}, GPUs {gpus}; tokenizers identical", flush=True)
    return len(gpus)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_push", action="store_true")
    args = ap.parse_args()
    n_gpu = check_env()
    log_dir = REPO / "logs" / "kaggle_v8"
    log_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    sh([sys.executable, "eval/test_v7.py"], env=dict(os.environ, PYTHONPATH=str(REPO / "eval")))

    sec, _ = run("logs_speed_v8", MODELS["1p5b"], "streaming_b256", ST, 0, log_dir, ["--limit", "1"])
    print(f"speed: 1.5B streaming ~{sec / 2:.0f} s/dialogue incl. loading; rough total "
          f"{len(CONFIGS) * 120 * sec / 2 / n_gpu / 3600 * 0.8:.1f} h (0.5B configs are faster)", flush=True)

    jobs, errors = queue.Queue(), []
    for c in CONFIGS:
        jobs.put(c)

    def worker(gpu):
        while not errors:
            try:
                out, model, name, a = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                sec, log = run(out, model, name, a, gpu, log_dir)
                recs = [json.loads(l) for l in (REPO / out / f"{name}_fp32.jsonl").open()]
                line = (f"{Path(out).name}/{name}: {sum(r['correct'] for r in recs)}/{len(recs)} correct "
                        f"(unconditional), {sec / 60:.0f} min on GPU {gpu}")
                print(line, flush=True)
                push([Path(out) / f"{name}_fp32.jsonl", log], f"v8 (Kaggle fp32): {line}", not args.no_push, lock)
            except Exception as e:
                errors.append(e)
                print(f"ERROR in {out}/{name}: {e}", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in range(n_gpu)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]

    sh([sys.executable, "eval/mechanism_v8.py", "--big", "results/logs_eval_v8_1p5b_fp32",
        "--small", "results/logs_eval_v8_0p5b_fp32", "--out", "results/mechanism_v8.md"],
       env=dict(os.environ, PYTHONPATH=str(REPO / "eval")))
    push(["results/mechanism_v8.md"], "v8 (Kaggle fp32): mechanism analysis", not args.no_push, lock)
    tar = Path("/kaggle/working/results_v8_npz.tar.gz") if Path("/kaggle/working").is_dir() else REPO / "results_v8_npz.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        for m in MODELS:
            t.add(REPO / f"results/logs_eval_v8_{m}_fp32", arcname=f"results/logs_eval_v8_{m}_fp32")
    print(f"done. .npz archive: {tar} ({tar.stat().st_size / 2**20:.0f} MB) - download it from the notebook output.")
    print((REPO / "results/mechanism_v8.md").read_text())


if __name__ == "__main__":
    main()
