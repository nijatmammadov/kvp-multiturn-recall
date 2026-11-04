"""Write Table 1 (recall of every configuration, conditioned on full-cache correct) as LaTeX rows.

usage (from report/): python table1.py .. data/table1.tex
"""
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) / "results"
OUT = Path(sys.argv[2])
KS = {4, 8}
COLS = {  # column -> (log dir, full-cache file)
    "w05": "logs_eval_v6k_fp32",
    "d05": "logs_eval_v6k_fp32_lead2",
    "w15": "logs_eval_v8_1p5b_fp32",
}
STREAM_W05 = "logs_eval_v8_0p5b_fp32"
ROWS = [  # (group, label, {col: (dir override or None, config)}, usable)
    ("ref", "Full cache", {c: (None, "full_b0") for c in COLS}, False),
    ("use", "StreamingLLM", {"w05": (STREAM_W05, "streaming_b256"), "w15": (None, "streaming_b256")}, True),
    ("use", "H2O", {"w05": (None, "h2o_b256"), "d05": (None, "h2o_b256")}, True),
    ("use", "KVP", {"w05": (None, "kvp_ft_b256"), "d05": (None, "kvp_ft_b256")}, True),
    ("use", "StreamingLLM + restore", {"w05": (STREAM_W05, "streaming_offload_b256"),
                                       "w15": (None, "streaming_offload_b256")}, True),
    ("use", "KVP + restore", {"w05": (None, "kvp_ft_offload_b256"), "d05": (None, "kvp_ft_offload_b256")}, True),
    ("diag", "StreamingLLM + answer tokens", {"w05": (STREAM_W05, "streaming_value_b256"),
                                              "w15": (None, "streaming_value_b256")}, False),
    ("diag", "StreamingLLM + answer + linking phrase", {"w05": (STREAM_W05, "streaming_keyval_b256"),
                                                        "w15": (None, "streaming_keyval_b256")}, False),
    ("diag", "StreamingLLM + whole user turn", {"w05": (STREAM_W05, "streaming_turn_b256"),
                                               "w15": (None, "streaming_turn_b256")}, False),
    ("diag", "KVP + answer tokens", {"w05": (None, "kvp_ft_heads48_b256"), "d05": (None, "kvp_ft_heads48_b256")}, False),
    ("diag", "KVP + answer + linking phrase", {"w05": (None, "kvp_ft_keyval_heads48_b256"),
                                               "d05": (None, "kvp_ft_keyval_heads48_b256")}, False),
    ("diag", "KVP + whole user turn, 6 of 48 heads", {"w05": (None, "kvp_ft_turn_heads6_b256")}, False),
    ("diag", "KVP + whole user turn, 12 of 48 heads", {"w05": (None, "kvp_ft_turn_heads12_b256")}, False),
    ("diag", "KVP + whole user turn, 24 of 48 heads", {"w05": (None, "kvp_ft_turn_heads24_b256")}, False),
    ("diag", "KVP + whole user turn, all 48 heads", {"w05": (None, "kvp_ft_turn_heads48_b256"),
                                                    "d05": (None, "kvp_ft_turn_heads48_b256")}, False),
]


def load(path):
    rows = {}
    for line in path.open():
        r = json.loads(line)
        if r["k"] in KS:
            rows[r["id"]] = r
    if not rows:
        raise SystemExit(f"no k = 4, 8 rows in {path}")
    return rows


def keep_set(d):
    return {i for i, r in load(ROOT / d / "full_b0_fp32.jsonl").items() if r["correct"]}


def cell(col, spec):
    d, cfg = spec
    d = d or COLS[col]
    keep = keep_set(COLS[col])
    if keep != keep_set(d):
        raise SystemExit(f"{d} and {COLS[col]} are conditioned on different dialogues")
    rows = load(ROOT / d / f"{cfg}_fp32.jsonl")
    missing = keep - rows.keys()
    if missing:
        raise SystemExit(f"{d}/{cfg}: {len(missing)} conditioned dialogues missing")
    c = sum(bool(rows[i]["correct"]) for i in keep)
    return c, len(keep)


def main():
    table = []
    for group, label, specs, usable in ROWS:
        vals = {col: cell(col, specs[col]) for col in COLS if col in specs}
        table.append((group, label, vals, usable))
    best = {}
    for col in COLS:
        cands = [v[col][0] / v[col][1] for g, _, v, u in table if u and col in v]
        best[col] = max(cands)

    lines, prev = [], None
    heads = {"ref": None, "use": r"\multicolumn{4}{@{}l}{\emph{Eviction methods (usable in a real chat)}}\\",
             "diag": r"\multicolumn{4}{@{}l}{\emph{Diagnostic: part of the fact force-kept, using its known position}}\\"}
    for group, label, vals, usable in table:
        if group != prev:
            if prev is not None:
                lines.append(r"\midrule")
            if heads[group]:
                lines.append(heads[group])
            prev = group
        cells = []
        for col in COLS:
            if col not in vals:
                cells.append("--")
                continue
            c, n = vals[col]
            s = f"{c}/{n} ({round(100 * c / n)}\\%)"
            if usable and c / n == best[col]:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        indent = r"\quad " if group != "ref" else ""
        lines.append(indent + label + " & " + " & ".join(cells) + r" \\")
    head = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r" & Words, 0.5B & Digits, 0.5B & Words, 1.5B \\", r"\midrule"]
    OUT.write_text("\n".join(head + lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    for g, l, v, u in table:
        print(f"{l:42s}", {c: f"{a}/{b}" for c, (a, b) in v.items()})


if __name__ == "__main__":
    main()
