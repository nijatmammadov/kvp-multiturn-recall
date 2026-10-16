"""Batch v7 follow-up: does retrieval still find the needle turn when the question is paraphrased? CPU only, no LLM.

For every dialogue of the word set (q1) and the digit lead-2 set, the final question is replaced by each paraphrase
template below (fixed before any result was computed). Retrievers pick one earlier user turn; hit = the needle turn.
  lexical: offload.choose_turn (the v7 retriever; no pick if fewer than 2 shared content words = miss)
  embed:   cosine similarity of sentence embeddings (sentence-transformers), argmax over earlier user turns

usage: python eval/paraphrase_check.py --retrievers lexical [embed] --out results/paraphrase_v7.md
"""
import argparse
import json
from pathlib import Path

from offload import choose_turn

RESULTS = Path(__file__).resolve().parent.parent / "results"
WORDS = {
    "original": None,
    "P1 name/where": "What's the name of my {owner}'s {animal}, and where does it live?",
    "P2 pet/town": "Do you remember what my {owner}'s pet is named and which town it's in?",
    "P3 belongs to": "Remind me: the {animal} that belongs to my {owner} - what's it called and where?",
    "P4 no entity": "Earlier I mentioned a pet. What was its name and home city?",
    "P5 has a": "My {owner} has a {animal}; tell me its name and the city it lives in.",
}
DIGITS = {
    "original": None,
    "P1 code for": "What's the code for the {key}?",
    "P2 get into": "I need to get into the {adj} {noun}. What were the digits?",
    "P3 noun only": "Which number did I give you to open the {noun}?",
    "P4 no entity": "What was the access code I told you earlier?",
    "P5 PIN": "Can you recall the PIN I mentioned?",
}


def questions(row, templates):
    out = {}
    for name, t in templates.items():
        if t is None:
            out[name] = row["user_turns"][-1]
        elif row.get("needle_type") == "words":
            owner, animal = row["key"].split("'s ")
            out[name] = t.format(owner=owner, animal=animal)
        else:
            adj, noun, code = row["key"].split(" ")
            out[name] = t.format(key=row["key"], adj=adj, noun=noun)
    return out


def lexical(past, q):
    return choose_turn(past, q)[0]


def make_embed(model_name):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")

    def pick(past, q):
        e = model.encode(past + [q], normalize_embeddings=True, convert_to_tensor=True)
        return int((e[:-1] @ e[-1]).argmax())

    return pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrievers", nargs="+", default=["lexical"], choices=["lexical", "embed"])
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pickers = {"lexical": lexical}
    if "embed" in args.retrievers:
        pickers["embed"] = make_embed(args.embed_model)
    pickers = {k: v for k, v in pickers.items() if k in args.retrievers}

    lines = ["# Retrieval under paraphrase (batch v7 follow-up)", "",
             "Hit = the retriever picks the needle turn among the earlier user turns at the final question "
             "(all k = 2, 4, 8; 180 dialogues per set). Templates fixed before results.", ""]
    for set_name, path, templates in (("words", "mt_niah_words_lead2_q1.jsonl", WORDS),
                                      ("digits", "mt_niah_lead2.jsonl", DIGITS)):
        rows = [json.loads(l) for l in (RESULTS / path).open()]
        lines += [f"## {set_name}", "", "| question | example | " + " | ".join(pickers) + " |",
                  "|---|---|" + "---|" * len(pickers)]
        for name in templates:
            cells = []
            for pname, pick in pickers.items():
                hits = 0
                for r in rows:
                    q = questions(r, {name: templates[name]})[name]
                    hits += pick(r["user_turns"][:-1], q) == r["needle_turn"]
                cells.append(f"{hits} / {len(rows)}")
            ex = questions(rows[0], {name: templates[name]})[name].replace("|", "/")
            lines.append(f"| {name} | {ex} | " + " | ".join(cells) + " |")
        lines.append("")
    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
