"""Run eval/paraphrase_check.py with the lexical and the embedding retriever (CPU is enough) and push the result.

usage (repo root, Kaggle): python scripts/kaggle_paraphrase.py [--no_push]
"""
import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_v6 import REPO, push  # noqa: E402

OUT = "results/paraphrase_v7.md"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_push", action="store_true")
    args = ap.parse_args()
    env = dict(os.environ, PYTHONPATH=str(REPO / "eval"))
    subprocess.run([sys.executable, "eval/paraphrase_check.py", "--retrievers", "lexical", "embed", "--out", OUT],
                   cwd=REPO, env=env, check=True)
    push([OUT], "v7 follow-up: retrieval under paraphrase, lexical vs embedding", not args.no_push, threading.Lock())


if __name__ == "__main__":
    main()
