"""CPU checks for the v5 word needle, run before any GPU time.
Needs the Qwen tokenizer and the OASST2 train split (downloaded on first run).
Run from eval/: uv run --project $P python test_word_needle.py
"""
import json
from functools import cache
from pathlib import Path

from transformers import AutoTokenizer

from evict_eval import MODEL, find_spans, score, turn_tokens
from mt_niah import CITY, K_VALUES, N_PER_K, PET, SYSTEM, TITLE, build_rows, load_filler

RESULTS = Path(__file__).resolve().parent.parent / "results"


@cache
def tok():
    return AutoTokenizer.from_pretrained(MODEL)


@cache
def filler():
    return load_filler()


def words_row(name="Professor Biscuit", city="Graz"):
    return {"needle_type": "words", "name": name, "city": city, "value": f"{name} | {city}"}


def test_no_city_inside_another():
    for a in CITY:
        for b in CITY:
            assert a == b or a.lower() not in b.lower(), (a, b)


def test_score_words():
    r = words_row()
    cases = {
        "Professor Biscuit, Graz": (True, True, True, False),
        "professor biscuit — GRAZ": (True, True, True, False),
        "Professor Biscuit": (False, True, False, False),
        "Captain Biscuit, Graz": (False, False, True, False),
        "Professor Biscuit, Graz or maybe Linz": (False, True, False, True),
        "Professor Biscuit, Linz": (False, True, False, True),
    }
    for answer, (correct, name_ok, city_ok, hedged) in cases.items():
        _, c, extra = score(r, answer)
        got = (c, extra["name_ok"], extra["city_ok"], extra["city_hedged"])
        assert got == (correct, name_ok, city_ok, hedged), (answer, got)


def test_score_digits_unchanged():
    r = {"value": "77013"}
    assert score(r, "77013")[:2] == ("77013", True)
    assert score(r, "The code is 77013.")[:2] == ("77013", True)
    assert score(r, "770134")[:2] == (None, False)
    assert score(r, "7701")[:2] == (None, False)


def test_find_spans_every_name_and_city():
    t = tok()
    for title in TITLE:
        for pet in PET:
            name = f"{title} {pet}"
            for city in CITY:
                needle = f"My friend's dog is called {name} and lives in {city}."
                row = {"system": SYSTEM, "user_turns": ["x", "y", needle]}
                ids = turn_tokens(t, row, 2)
                n_span, c_span = find_spans(t, ids, [name]), find_spans(t, ids, [city])
                for text, span in ((name, n_span), (city, c_span)):
                    assert span == list(range(span[0], span[-1] + 1)), (text, span)
                    assert t.decode([ids[i] for i in span]).strip() == text, (text, t.decode([ids[i] for i in span]))
                assert find_spans(t, ids, [name, city]) == sorted(set(n_span + c_span))


def test_committed_sets_reproduce():
    for seed, lead, needle, path in ((0, 0, "digits", "mt_niah.jsonl"), (2, 2, "digits", "mt_niah_lead2.jsonl"),
                                     (4, 2, "words", "mt_niah_words_lead2_q1.jsonl")):
        want = (RESULTS / path).read_text().splitlines()
        got = [json.dumps(r) for r in build_rows(filler(), seed, lead=lead, needle_type=needle)]
        assert got == want, f"{path}: regenerated set differs from the committed one"


def test_word_set():
    rows = build_rows(filler(), 4, lead=2, needle_type="words")
    assert rows == build_rows(filler(), 4, lead=2, needle_type="words"), "not deterministic"
    assert len(rows) == len(K_VALUES) * N_PER_K
    assert len({r["id"] for r in rows}) == len(rows)
    for k in K_VALUES:
        assert sum(r["k"] == k for r in rows) == N_PER_K
    for r in rows:
        turns = r["user_turns"]
        assert r["needle_turn"] == 2 and turns[2] == r["needle"], r["id"]
        assert len(turns) == 2 + 1 + r["k"] + 1, r["id"]
        assert r["city"] in CITY and r["value"] == f"{r['name']} | {r['city']}", r["id"]
        banned = [w.lower() for w in (*r["name"].split(), r["city"])]
        for i, t in enumerate(turns):
            if i != 2:
                assert not any(w in t.lower() for w in banned), (r["id"], i)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
