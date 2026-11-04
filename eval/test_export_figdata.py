import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_figdata import export_dialogue  # noqa: E402


class Tok:
    def __init__(self):
        self.vocab = []

    def ids(self, pieces):
        out = []
        for p in pieces:
            if p not in self.vocab:
                self.vocab.append(p)
            out.append(self.vocab.index(p))
        return out

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)


def build():
    tok = Tok()
    turns = ["How do I learn?", "Tell me more.", "My cousin's hamster is called Lady Waffle and lives in Bergen.",
             "What is my cousin's hamster called, and in which city does it live?"]
    row = {"user_turns": turns, "needle_turn": 2, "key": "cousin's hamster", "name": "Lady Waffle", "city": "Bergen"}
    stream = ["<|im_start|>", "system", "\n", "You", " are", " helpful", ".", "<|im_end|>", "\n"]
    fact_pos = None
    for i, t in enumerate(turns):
        if i:
            stream += ["<|im_end|>", "\n"]
        stream += ["<|im_start|>", "user", "\n"]
        words = t.replace("'s", " 's").split(" ")
        start = len(stream)
        pieces = [w if j == 0 else " " + w for j, w in enumerate(words)]
        pieces = [p.replace(" 's", "'s") for p in pieces]
        stream += pieces
        if i == 2:
            fact_pos = list(range(start, len(stream)))
        stream += ["<|im_end|>", "\n", "<|im_start|>", "assistant", "\n"]
        if i < len(turns) - 1:
            stream += ["Sure", "."]
    ids = tok.ids(stream)
    assert tok.decode(ids) == "".join(stream)
    return tok, row, ids, fact_pos


def main():
    tok, row, ids, fact_pos = build()
    im_start, im_end = tok.vocab.index("<|im_start|>"), tok.vocab.index("<|im_end|>")
    n_l, n_h, b = 2, 2, 64
    kept = np.full((3, n_l, n_h, b), -1, dtype=np.int32)
    # head (0,0) keeps the whole fact turn; head (1,1) keeps only fact token 0; others keep nothing of the fact
    kept[-1, 0, 0, : len(fact_pos)] = fact_pos
    kept[-1, 1, 1, 0] = fact_pos[0]
    events = np.array(["t0_prefill", "t2_prefill", "t3_prefill"])
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.npz"
        np.savez_compressed(p, events=events, stream=np.array(ids, dtype=np.int32), kept=kept)
        rec = {"answer": "Bella", "correct": False}
        out = export_dialogue(p, row, rec, tok, im_start, im_end, 4)
        f = out["fact"]
        assert "".join(f["pieces"]) == row["user_turns"][2], f["pieces"]
        assert f["counts"][0] == 2 and all(c == 1 for c in f["counts"][1:]), f["counts"]
        assert len(f["matrix"]) == 4 and f["matrix"][0] == [1] * len(fact_pos) and f["matrix"][3][0] == 1
        roles = dict(zip(f["pieces"], f["roles"]))
        assert roles[" Lady"] == roles[" Waffle"] == roles[" Bergen."] == "answer", roles
        assert roles[" cousin"] == roles["'s"] == roles[" hamster"] == "link" and roles[" called"] == "", roles
        assert "".join(out["question"]["pieces"]) == row["user_turns"][3]
        assert out["question"]["counts"] == [0] * len(out["question"]["pieces"])

        np.savez_compressed(p, events=events, stream=np.array(ids, dtype=np.int32))
        full = export_dialogue(p, row, {"answer": "Lady Waffle, Bergen", "correct": True}, tok, im_start, im_end, 4)
        assert full["fact"]["matrix"] is None and set(full["fact"]["counts"]) == {4}

        bad = np.array(["t0_prefill", "t2_end"])
        np.savez_compressed(p, events=bad, stream=np.array(ids, dtype=np.int32), kept=kept[:2])
        try:
            export_dialogue(p, row, rec, tok, im_start, im_end, 4)
        except RuntimeError as e:
            assert "last event" in str(e)
        else:
            raise AssertionError("wrong last event not caught")
    print("export_figdata: all tests passed")


if __name__ == "__main__":
    main()
