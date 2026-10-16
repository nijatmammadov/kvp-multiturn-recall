"""Build the short RULER-style recall prompts used to fine-tune the KVP agents (batch v4). Pure Python, no model.

  python eval/make_recall_prompts.py --wikitext path/to/wikitext-2/train.txt --out data/recall_prompts.jsonl

Haystack: WikiText-2 train (CC BY-SA 3.0), tokenized release as shipped in pytorch/examples
(word_language_model/data/wikitext-2/train.txt); sentences containing <unk> are dropped and spacing is detokenized.
Each prompt: an instruction, Wikipedia sentences with "special magic {type}" needles inserted at random depths,
and a question; the assistant turn is pre-filled with an answer prefix so Qwen's generated tokens are the value(s).
Variants follow RULER's NIAH family: single, multikey (3 decoy needles never asked), multivalue (3 values, one key),
multiquery (2 keys asked). Values: 7-digit numbers, UUIDs or words. Held out from MT-NIAH (the eval benchmark):
different wording, key vocabulary, value format, filler source and a single turn instead of a dialogue.
Length is controlled in characters (~4 chars per Qwen token for this text); the server-side dataset class
re-tokenizes and checks the real token counts.
"""
import argparse
import json
import random
import re
import uuid

SYSTEM = "You are a helpful assistant."
KEY_ADJ = ["quiet", "bright", "ancient", "rapid", "gentle", "bitter", "curious", "distant", "eager", "fragile",
           "humble", "jolly", "loyal", "nimble", "proud", "sturdy", "tender", "vivid", "wise", "zealous"]
KEY_NOUN = ["falcon", "meadow", "compass", "violin", "glacier", "pepper", "saddle", "thimble", "walnut", "beacon",
            "canyon", "dolphin", "ember", "feather", "harvest", "island", "kettle", "marble", "nectar", "otter"]
VALUE_WORDS = ["apricot", "basalt", "cobalt", "dynamo", "eclipse", "fjord", "gazebo", "hydrant", "igloo", "jigsaw",
               "kayak", "lagoon", "mosaic", "nebula", "oasis", "pylon", "quartz", "rhubarb", "sonnet", "tundra"]
MT_NIAH_WORDS = {"amber", "silent", "northern", "crimson", "hollow", "iron", "velvet", "frozen", "golden", "hidden",
                 "lunar", "rusty", "copper", "misty", "broken", "scarlet", "vault", "archive", "harbor", "tower",
                 "garden", "locker", "bridge", "cellar", "station", "gallery", "orchard", "foundry", "lantern",
                 "chapel", "depot", "mill"}
TYPES = {"numbers": ("number", "numbers"), "uuids": ("uuid", "uuids"), "words": ("word", "words")}
VARIANTS = (("single", 0.40), ("multikey", 0.30), ("multivalue", 0.15), ("multiquery", 0.15))
CHARS_PER_TOKEN = 4.0


def detok(s):
    for a, b in ((" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", "."), (" ,", ","), (" .", "."), (" ;", ";"),
                 (" :", ":"), (" 's", "'s"), (" n't", "n't"), ("( ", "("), (" )", ")"), (' " ', ' "')):
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def load_sentences(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("="):
            continue
        for sent in line.split(" . "):
            if "<unk>" in sent or len(sent) < 60:
                continue
            sent = detok(sent)
            out.append(sent if sent.endswith(".") else sent + ".")
    if len(out) < 10000:
        raise SystemExit(f"only {len(out)} usable sentences in {path}")
    return out


def value(rng, vtype):
    if vtype == "numbers":
        return str(rng.randint(1_000_000, 9_999_999))
    if vtype == "uuids":
        return str(uuid.UUID(int=rng.getrandbits(128), version=4))
    return rng.choice(VALUE_WORDS)


def make(rng, sents, idx, min_tokens, max_tokens):
    variant = rng.choices([v for v, _ in VARIANTS], weights=[w for _, w in VARIANTS])[0]
    vtype = rng.choices(list(TYPES), weights=[0.6, 0.2, 0.2])[0]
    one, many = TYPES[vtype]
    keys = rng.sample([f"{a}-{n}" for a in KEY_ADJ for n in KEY_NOUN], 4)
    vals = []
    while len(vals) < 4:
        v = value(rng, vtype)
        if v not in vals:
            vals.append(v)
    if variant == "single":
        needles, asked = [(keys[0], vals[0])], [keys[0]]
    elif variant == "multikey":
        needles, asked = list(zip(keys, vals)), [keys[0]]
    elif variant == "multivalue":
        needles, asked = [(keys[0], v) for v in vals[:3]], [keys[0]]
    else:
        needles, asked = [(keys[0], vals[0]), (keys[1], vals[1])], keys[:2]
    rng.shuffle(needles)
    answers = [v for k, v in needles if k in asked]

    who = " and ".join(asked)
    if len(answers) == 1:
        question = f"What is the special magic {one} for {who} mentioned in the provided text?"
        prefix = f"The special magic {one} for {who} mentioned in the provided text is"
    else:
        question = f"What are all the special magic {many} for {who} mentioned in the provided text?"
        prefix = f"The special magic {many} for {who} mentioned in the provided text are"
    intro = (f"Some special magic {many} are hidden within the following text. Make sure to memorize them. "
             f"I will quiz you about the {many} afterwards.")
    needle_text = [f"One of the special magic {many} for {k} is: {v}." for k, v in needles]

    target_chars = rng.randint(min_tokens, max_tokens) * CHARS_PER_TOKEN
    used = len(intro) + len(question) + len(prefix) + sum(len(t) + 1 for t in needle_text)
    hay, i = [], rng.randrange(len(sents))
    while used < target_chars:
        hay.append(sents[i % len(sents)])
        used += len(hay[-1]) + 1
        i += 1
    slots = sorted(rng.randint(0, len(hay)) for _ in needles)
    for j, (slot, t) in enumerate(zip(slots, needle_text)):
        hay.insert(slot + j, t)
    user = f"{intro}\n{' '.join(hay)}\n{question}"
    for t, (k, v) in zip(needle_text, needles):
        if user.count(t) != 1:
            raise RuntimeError(f"needle {k}={v} not inserted exactly once")
    return {"id": idx, "variant": variant, "vtype": vtype, "system": SYSTEM, "user": user,
            "answer_prefix": prefix, "answers": answers, "asked": asked,
            "decoys": [k for k, _ in needles if k not in asked]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wikitext", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--min_tokens", type=int, default=500)
    ap.add_argument("--max_tokens", type=int, default=1400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sents = load_sentences(args.wikitext)
    rng = random.Random(args.seed)
    rows = [make(rng, sents, i, args.min_tokens, args.max_tokens) for i in range(args.n)]
    for r in rows:
        if set(re.findall(r"[a-z]+", " ".join(r["asked"] + r["decoys"]))) & MT_NIAH_WORDS:
            raise RuntimeError(f"row {r['id']}: key overlaps MT-NIAH vocabulary")
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    lens = [len(r["user"]) / CHARS_PER_TOKEN for r in rows]
    print(f"wrote {len(rows)} prompts to {args.out} from {len(sents)} sentences; "
          f"est. tokens min/mean/max {min(lens):.0f}/{sum(lens) / len(lens):.0f}/{max(lens):.0f}")
    print("variants", dict(Counter(r["variant"] for r in rows)), "types", dict(Counter(r["vtype"] for r in rows)))


if __name__ == "__main__":
    main()
