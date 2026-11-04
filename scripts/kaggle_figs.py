"""Data for the report figures, on Kaggle (fp32, words q1, k in {4, 8}).

Part 1 (~30 min on T4 x2): b = 256, the first 10 dialogues per k, with .npz logs, into results/logs_figs_fp32 (not
  committed): full, kvp_ft, h2o, kvp_ft + answer tokens in all heads, kvp_ft + answer + linking phrase in all heads,
  kvp_ft + restore. Regression: answers must match the committed fp32 runs (>= 90% identical per config). Then
  eval/export_figdata.py writes results/figdata.json (token head counts for the token-highlight and heads x tokens
  figures), which is committed and pushed. Skipped when results/figdata.json already exists (use --redo_figs).
Part 2 (--sweep, ~3.5 h on T4 x2): budgets 64, 128, 512 for streaming, h2o, kvp_ft, and 512 for kvp_ft + restore,
  on all 120 dialogues, into results/logs_sweep_fp32; each config's JSONL is pushed when it finishes (b = 256 already
  exists in logs_eval_v6k_fp32 and logs_eval_v8_0p5b_fp32). Restore needs the whole matched user turn plus the 20
  always-kept rows to fit in the budget; user turns reach ~120 tokens, so it only runs at b >= 256 (at b = 64 it
  stopped with "protected rows exceed cache size"). A failing sweep config is reported and the others still run; the
  script exits non-zero at the end. evict_eval.py resumes by dialogue id, so a rerun continues.

usage (repo root): python scripts/kaggle_figs.py --apple /kaggle/working/ml-learning-to-evict [--sweep] [--no_push]
"""
import argparse
import queue
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_v6 import KVP, REPO, check_env, env_for, push, sh  # noqa: E402
from kaggle_v6b import OUT_W, WORDS, answers, run  # noqa: E402

FIG_OUT = "results/logs_figs_fp32"
FIG_JSON = "results/figdata.json"
SWEEP_OUT = "results/logs_sweep_fp32"
AGENTS_ARGS = ["--agents_dir", KVP[KVP.index("--agents_dir") + 1]]
P48 = ["--protect_heads", "48"]
FIG_CONFIGS = [  # (name, args, committed run to compare answers with)
    ("full_b0", ["--method", "full", "--budget", "0"], f"{OUT_W}/full_b0_fp32.jsonl"),
    ("kvp_ft_b256", KVP + ["--tag", "ft"], f"{OUT_W}/kvp_ft_b256_fp32.jsonl"),
    ("h2o_b256", ["--method", "h2o", "--budget", "256"], f"{OUT_W}/h2o_b256_fp32.jsonl"),
    ("kvp_ft_heads48_b256", KVP + ["--protect_needle", "value"] + P48 + ["--tag", "ft_heads48"],
     f"{OUT_W}/kvp_ft_heads48_b256_fp32.jsonl"),
    ("kvp_ft_keyval_heads48_b256", KVP + ["--protect_needle", "keyvalue"] + P48 + ["--tag", "ft_keyval_heads48"],
     f"{OUT_W}/kvp_ft_keyval_heads48_b256_fp32.jsonl"),
    ("kvp_ft_offload_b256", KVP + ["--offload_retrieve", "--tag", "ft_offload"], f"{OUT_W}/kvp_ft_offload_b256_fp32.jsonl"),
]
FIG_LIMIT = ["--limit", "10"]
RESTORE_MIN_BUDGET = 256


def sweep_configs():
    out = []
    for b in (64, 128, 512):
        kvp = ["--method", "kvp", "--budget", str(b)] + AGENTS_ARGS
        out += [
            (f"kvp_ft_b{b}", kvp + ["--tag", "ft"]),
            (f"h2o_b{b}", ["--method", "h2o", "--budget", str(b)]),
            (f"streaming_b{b}", ["--method", "streaming", "--budget", str(b)]),
        ]
        if b >= RESTORE_MIN_BUDGET:
            out.append((f"kvp_ft_offload_b{b}", kvp + ["--offload_retrieve", "--tag", "ft_offload"]))
    return out


def run_all(jobs_list, out, apple, n_gpu, log_dir, extra=(), on_done=None, keep_going=False):
    jobs, errors = queue.Queue(), []
    for j in jobs_list:
        jobs.put(j)

    def worker(gpu):
        while keep_going or not errors:
            try:
                name, a = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                sec, log = run(name, WORDS, out, a, apple, gpu, log_dir, extra)
                recs = answers(f"{out}/{name}_fp32.jsonl")
                line = (f"{Path(out).name}/{name}: {sum(r['correct'] for r in recs.values())}/{len(recs)} correct "
                        f"(unconditional), {sec / 60:.0f} min on GPU {gpu}")
                print(line, flush=True)
                if on_done:
                    on_done(name, log, line)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apple", required=True)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--redo_figs", action="store_true")
    ap.add_argument("--no_push", action="store_true")
    args = ap.parse_args()
    n_gpu = check_env(args.apple)
    log_dir = REPO / "logs" / "kaggle_figs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    sh([sys.executable, "eval/test_v6.py"], env=env_for(args.apple))
    sh([sys.executable, "eval/test_v7.py"], env=env_for(args.apple))
    sh([sys.executable, "eval/test_export_figdata.py"])

    if (REPO / FIG_JSON).exists() and not args.redo_figs:
        print(f"{FIG_JSON} exists: skipping part 1 (use --redo_figs to rebuild it)", flush=True)
    else:
        run_all([(n, a) for n, a, _ in FIG_CONFIGS], FIG_OUT, args.apple, n_gpu, log_dir, FIG_LIMIT)
        for name, _, ref in FIG_CONFIGS:
            new, old = answers(f"{FIG_OUT}/{name}_fp32.jsonl"), answers(ref)
            same = sum(new[i]["answer"] == old[i]["answer"] for i in new)
            print(f"regression {name}: {same}/{len(new)} answers identical to {ref}", flush=True)
            if same < 0.9 * len(new):
                sys.exit(f"REGRESSION FAILED for {name}: this session does not reproduce the committed fp32 runs")
        sh([sys.executable, "eval/export_figdata.py", "--logs", FIG_OUT, "--data", WORDS, "--out", FIG_JSON])
        push([FIG_JSON, *sorted(log_dir.glob(f"{Path(FIG_OUT).name}__*.log"))],
             "figures (Kaggle fp32): figdata.json for the token-highlight and heads x tokens figures",
             not args.no_push, lock)

    if args.sweep:
        def done(name, log, line):
            push([Path(SWEEP_OUT) / f"{name}_fp32.jsonl", log], f"budget sweep (Kaggle fp32): {line}",
                 not args.no_push, lock)

        run_all(sweep_configs(), SWEEP_OUT, args.apple, n_gpu, log_dir, on_done=done, keep_going=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
