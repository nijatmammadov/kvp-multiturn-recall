"""Batch v7 on Kaggle (fp32): offload + retrieve on top of kvp_ft, against kvp_ft and H2O.

Configs (b = 256, k in {4, 8}, fp32):
  words q1 (results/logs_eval_v6k_fp32, pairs with the fp32 ladder): kvp_ft_offload, h2o
  digits lead-2 (results/logs_eval_v6k_fp32_lead2): kvp_ft, kvp_ft_offload, h2o
Regression check first (10 kvp_ft word dialogues >= 9/10 identical to the fp32 ladder), as in v6b.

usage (repo root): python scripts/kaggle_v7.py --apple /kaggle/working/ml-learning-to-evict [--no_push]
"""
import argparse
import queue
import sys
import tarfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_v6 import KVP, REPO, check_env, env_for, push, sh  # noqa: E402
from kaggle_v6b import DIGITS, OUT_D, OUT_W, WORDS, answers, run  # noqa: E402

H2O = ["--method", "h2o", "--budget", "256"]
OFF = KVP + ["--offload_retrieve", "--tag", "ft_offload"]
CONFIGS = [
    ("kvp_ft_offload_b256", WORDS, OUT_W, OFF),
    ("kvp_ft_offload_b256", DIGITS, OUT_D, OFF),
    ("h2o_b256", WORDS, OUT_W, H2O),
    ("kvp_ft_b256", DIGITS, OUT_D, KVP + ["--tag", "ft"]),
    ("h2o_b256", DIGITS, OUT_D, H2O),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apple", required=True)
    ap.add_argument("--no_push", action="store_true")
    args = ap.parse_args()
    n_gpu = check_env(args.apple)
    log_dir = REPO / "logs" / "kaggle_v7"
    log_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    sh([sys.executable, "eval/test_v6.py"], env=env_for(args.apple))
    sh([sys.executable, "eval/test_v7.py"], env=env_for(args.apple))

    reg = "logs_regress_v7"
    sec, _ = run("kvp_ft_b256", WORDS, reg, KVP + ["--tag", "ft"], args.apple, 0, log_dir, ["--limit", "5"])
    new, old = answers(f"{reg}/kvp_ft_b256_fp32.jsonl"), answers(f"{OUT_W}/kvp_ft_b256_fp32.jsonl")
    same = sum(new[i]["answer"] == old[i]["answer"] for i in new)
    print(f"regression: {same}/{len(new)} kvp_ft answers identical to the fp32 ladder; estimated total "
          f"{len(CONFIGS) * 120 * sec / len(new) / n_gpu / 3600:.1f} h (H2O may be slower)", flush=True)
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
                push([Path(out) / f"{name}_fp32.jsonl", log], f"v7 (Kaggle fp32): {line}", not args.no_push, lock)
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

    sh([sys.executable, "eval/offload_v7.py", "--words", OUT_W, "--digits", OUT_D, "--out", "results/offload_v7.md"])
    push(["results/offload_v7.md"], "v7 (Kaggle fp32): offload analysis", not args.no_push, lock)

    tar = Path("/kaggle/working/results_v7_npz.tar.gz") if Path("/kaggle/working").is_dir() else REPO / "results_v7_npz.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        for name, _, out, _ in CONFIGS:
            t.add(REPO / out / f"{name}_fp32", arcname=f"{out}/{name}_fp32")
    print(f"done. .npz archive: {tar} ({tar.stat().st_size / 2**20:.0f} MB) - download it from the notebook output.")
    print((REPO / "results/offload_v7.md").read_text())


if __name__ == "__main__":
    main()
