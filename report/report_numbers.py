import json
import math
import sys
from pathlib import Path

from scipy.stats import binomtest

R = Path(sys.argv[1]) / "results"
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
KS = {4, 8}


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["k"] in KS:
                rows[r["id"]] = r
    if not rows:
        raise SystemExit(f"no k=4,8 rows in {path}")
    return rows


def wilson(c, n, z=1.959964):
    p = c / n
    d = 1 + z * z / n
    m = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, m - h), min(1.0, m + h)


def cond(rows, keep):
    missing = keep - rows.keys()
    if missing:
        raise SystemExit(f"{len(missing)} conditioned ids missing")
    return {i: rows[i] for i in keep}


def acc(rows):
    c = sum(bool(r["correct"]) for r in rows.values())
    n = len(rows)
    lo, hi = wilson(c, n)
    return c, n, lo, hi


def mcnemar(a, b):
    if a.keys() != b.keys():
        raise SystemExit("mcnemar on different dialogue sets")
    only_a = sum(a[i]["correct"] and not b[i]["correct"] for i in a)
    only_b = sum(b[i]["correct"] and not a[i]["correct"] for i in a)
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return only_a, only_b, p


def mean(rows, key):
    v = [r[key] for r in rows.values() if r.get(key) is not None]
    if not v:
        return None
    return sum(v) / len(v)


sets = {
    "words05": R / "logs_eval_v6k_fp32",
    "digits05": R / "logs_eval_v6k_fp32_lead2",
    "s05": R / "logs_eval_v8_0p5b_fp32",
    "s15": R / "logs_eval_v8_1p5b_fp32",
}
data = {}
for name, d in sets.items():
    full = load(d / "full_b0_fp32.jsonl")
    keep = {i for i, r in full.items() if r["correct"]}
    data[name] = {}
    for f in sorted(d.glob("*_fp32.jsonl")):
        cfg = f.name[: -len("_fp32.jsonl")]
        data[name][cfg] = cond(load(f), keep)
    print(f"{name}: full cache {len(keep)}/{len(full)}; configs {sorted(data[name])}")

print()
for name in data:
    for cfg, rows in data[name].items():
        c, n, lo, hi = acc(rows)
        print(f"{name:9s} {cfg:34s} {c:3d}/{n:3d} [{lo:.3f},{hi:.3f}] surv={mean(rows,'needle_at_answer')} whole={mean(rows,'needle_heads_full_at_answer')} mem={mean(rows,'peak_mem_mb'):.0f} s={mean(rows,'seconds'):.2f}")

print()
pairs = [
    ("words05", "kvp_ft_offload_b256", "kvp_ft_b256"),
    ("digits05", "kvp_ft_offload_b256", "kvp_ft_b256"),
    ("words05", "h2o_b256", "kvp_ft_b256"),
    ("digits05", "h2o_b256", "kvp_ft_b256"),
    ("words05", "kvp_ft_offload_b256", "h2o_b256"),
    ("digits05", "kvp_ft_offload_b256", "h2o_b256"),
    ("words05", "kvp_ft_keyval_heads48_b256", "kvp_ft_heads48_b256"),
    ("words05", "kvp_ft_turn_heads48_b256", "kvp_ft_keyval_heads48_b256"),
    ("s15", "streaming_keyval_b256", "streaming_value_b256"),
    ("s05", "streaming_keyval_b256", "streaming_value_b256"),
]
for s, a, b in pairs:
    if a in data[s] and b in data[s]:
        print(s, a, "vs", b, mcnemar(data[s][a], data[s][b]))
    else:
        print("MISSING", s, a, b)
