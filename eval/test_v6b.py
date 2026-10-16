"""Tokenizer checks for --protect_needle keyvalue (batch v6, amendment 3). Needs the Qwen tokenizer (downloads it).
Run from eval/: python test_v6b.py
"""
import json
from pathlib import Path

from transformers import AutoTokenizer

from evict_eval import MODEL, key_positions, needle_positions, turn_tokens

RESULTS = Path(__file__).resolve().parent.parent / "results"
SETS = ("mt_niah_words_lead2_q1.jsonl", "mt_niah_lead2.jsonl")


def test_key_spans():
    tok = AutoTokenizer.from_pretrained(MODEL)
    for name in SETS:
        rows = [json.loads(l) for l in (RESULTS / name).open()]
        for r in rows:
            ids = turn_tokens(tok, r, r["needle_turn"])
            kp = key_positions(tok, ids, r, 0)
            vp = needle_positions(tok, ids, r)
            assert kp == list(range(kp[0], kp[-1] + 1)), (r["id"], kp)
            assert tok.decode([ids[i] for i in kp]).strip() == r["key"], (r["id"], tok.decode([ids[i] for i in kp]))
            assert not set(kp) & set(vp), (r["id"], "key and value overlap")
            assert max(kp) < min(vp), (r["id"], "key must come before the value in the needle")
        print(f"PASS {name}: {len(rows)} key spans")


if __name__ == "__main__":
    test_key_spans()
