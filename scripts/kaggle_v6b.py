"""Batch v6 amendment 3 on Kaggle (fp32): which part of the needle turn must survive.

Configs (kvp_ft agents, b = 256, protection in all 48 heads, k in {4, 8}, fp32):
  words q1  (results/logs_eval_v6k_fp32, pairs with the existing fp32 ladder): value + linking phrase (keyvalue)
  digits lead-2 (results/logs_eval_v6k_fp32_lead2): full cache, value, keyvalue, whole turn
First a regression check: 10 kvp_ft word dialogues must reproduce the existing fp32 ladder answers (>= 9/10),
otherwise this session's numerics differ and the new words config would not pair with the old ones.

usage (repo root): python scripts/kaggle_v6b.py --apple /kaggle/working/ml-learning-to-evict [--no_push]
"""
import argparse
import json
import queue
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_v6 import AGENTS, KVP, REPO, check_env, env_for, push, sh  # noqa: E402

WORDS, DIGITS = "results/mt_niah_words_lead2_q1.jsonl", "results/mt_niah_lead2.jsonl"
OUT_W, OUT_D = "results/logs_eval_v6k_fp32", "results/logs_eval_v6k_fp32_lead2"
P48 = ["--protect_heads", "48"]
CONFIGS = [  # (name, data, out, args)
    ("kvp_ft_keyval_heads48_b256", WORDS, OUT_W, KVP + ["--protect_needle", "keyvalue"] + P48 + ["--tag", "ft_keyval_heads48"]),
    ("full_b0", DIGITS, OUT_D, ["--method", "full", "--budget", "0"]),
    ("kvp_ft_turn_heads48_b256", DIGITS, OUT_D, KVP + ["--protect_needle", "turn"] + P48 + ["--tag", "ft_turn_heads48"]),
    ("kvp_ft_heads48_b256", DIGITS, OUT_D, KVP + ["--protect_needle", "value"] + P48 + ["--tag", "ft_heads48"]),
    ("kvp_ft_keyval_heads48_b256", DIGITS, OUT_D, KVP + ["--protect_needle", "keyvalue"] + P48 + ["--tag", "ft_keyval_heads48"]),
]


def run(name, data, out, a, apple, gpu, log_dir, extra=()):
    log = log_dir / f"{Path(out).name}__{name}.log"
    cmd = [sys.executable, "eval/evict_eval.py", "--data", data, "--out_dir", out, "--dtype", "fp32", "--ks", "4,8",
           *a, *extra]
    t0 = time.time()
    with log.open("a") as f:
        f.write(f"$ {' '.join(cmd)}  (GPU {gpu})\n")
        f.flush()
        r = subprocess.run(cmd, cwd=REPO, env=env_for(apple, gpu), stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"{out}/{name} failed (exit {r.returncode}); see {log}")
    return time.time() - t0, log


def answers(path):
    return {json.loads(l)["id"]: json.loads(l) for l in (REPO / path).open()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apple", required=True)
    ap.add_argument("--no_push", action="store_true")
    args = ap.parse_args()
    n_gpu = check_env(args.apple)
    log_dir = REPO / "logs" / "kaggle_v6b"
    log_dir.mkdir(parents=True, exist_ok=True)
    for d in (OUT_W, OUT_D):
        (REPO / d).mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    sh([sys.executable, "eval/test_v6.py"], env=env_for(args.apple))
    sh([sys.executable, "eval/test_v6b.py"], env=env_for(args.apple))

    reg = "logs_regress_v6b"
    sec, _ = run("kvp_ft_b256", WORDS, reg, KVP + ["--tag", "ft"], args.apple, 0, log_dir, ["--limit", "5"])
    new, old = answers(f"{reg}/kvp_ft_b256_fp32.jsonl"), answers(f"{OUT_W}/kvp_ft_b256_fp32.jsonl")
    same = sum(new[i]["answer"] == old[i]["answer"] for i in new)
    print(f"regression: {same}/{len(new)} kvp_ft answers identical to the fp32 ladder ({sec / len(new):.0f} s/dialogue "
          f"incl. loading); estimated total {len(CONFIGS) * 120 * sec / len(new) / n_gpu / 3600:.1f} h", flush=True)
    if same < len(new) - 1:
        sys.exit("REGRESSION FAILED: this session does not reproduce the fp32 ladder; stop and report")

    jobs, errors = queue.Queue(), []
    for c in CONFIGS:
        jobs.put(c)

    def worker(gpu):
        while not errors:
            try:
                name, data, out, a = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                sec, log = run(name, data, out, a, args.apple, gpu, log_dir)
                recs = answers(f"{out}/{name}_fp32.jsonl")
                line = (f"{Path(out).name}/{name}: {sum(r['correct'] for r in recs.values())}/{len(recs)} correct "
                        f"(unconditional), {sec / 60:.0f} min on GPU {gpu}")
                print(line, flush=True)
                push([Path(out) / f"{name}_fp32.jsonl", log], f"v6b (Kaggle fp32): {line}", not args.no_push, lock)
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

    sh([sys.executable, "eval/binding_v6b.py", "--words", OUT_W, "--digits", OUT_D, "--out", "results/binding_v6b.md"])
    sh([sys.executable, "eval/paired_stats.py", "--logs", OUT_D, "--condition_on", "full:0", "--pairs",
        "kvp_ft_turn_heads48:kvp_ft_heads48", "kvp_ft_keyval_heads48:kvp_ft_heads48",
        "kvp_ft_turn_heads48:kvp_ft_keyval_heads48", "--pool_ks", "4,8", "--out", "results/paired_stats_v6b_digits.md"])
    push(["results/binding_v6b.md", "results/paired_stats_v6b_digits.md"], "v6b (Kaggle fp32): binding analysis",
         not args.no_push, lock)

    tar = Path("/kaggle/working/results_v6b_npz.tar.gz") if Path("/kaggle/working").is_dir() else REPO / "results_v6b_npz.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(REPO / OUT_D, arcname=OUT_D)
        t.add(REPO / OUT_W / "kvp_ft_keyval_heads48_b256_fp32", arcname=f"{OUT_W}/kvp_ft_keyval_heads48_b256_fp32")
    print(f"done. .npz archive: {tar} ({tar.stat().st_size / 2**20:.0f} MB) - download it from the notebook output.")
    print((REPO / "results/binding_v6b.md").read_text())


if __name__ == "__main__":
    main()
