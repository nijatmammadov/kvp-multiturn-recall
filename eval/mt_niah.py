import argparse
import json
import random
import re

K_VALUES = (2, 4, 8)
N_PER_K = 60
SEED = 0
SYSTEM = "You are a helpful assistant."
ADJ = ["amber", "silent", "northern", "crimson", "hollow", "iron", "velvet", "frozen",
       "golden", "hidden", "lunar", "rusty", "copper", "misty", "broken", "scarlet"]
NOUN = ["vault", "archive", "harbor", "tower", "garden", "locker", "bridge", "cellar",
        "station", "gallery", "orchard", "foundry", "lantern", "chapel", "depot", "mill"]
OWNER = ["friend", "neighbour", "sister", "colleague", "cousin", "teacher", "landlord", "brother"]
ANIMAL = ["dog", "cat", "parrot", "horse", "rabbit", "tortoise", "goat", "hamster"]
TITLE = ["Professor", "Captain", "Doctor", "Baron", "Major", "Lady", "Sir", "Madame"]
PET = ["Biscuit", "Pickles", "Waffle", "Noodle", "Pumpkin", "Marbles", "Truffle", "Crumpet",
       "Sprocket", "Tumbleweed", "Gingersnap", "Dumpling", "Pretzel", "Snickerdoodle", "Brambles", "Teacup"]
CITY = ["Graz", "Linz", "Ghent", "Brno", "Tartu", "Lund", "Aarhus", "Bergen", "Utrecht", "Leuven",
        "Porto", "Bologna", "Trieste", "Ljubljana", "Basel", "Innsbruck", "Tampere", "Aalborg", "Seville", "Nantes"]


def load_filler():
    from datasets import load_dataset

    ds = load_dataset("OpenAssistant/oasst2", split="train")
    turns = {
        t for t, role, lang in zip(ds["text"], ds["role"], ds["lang"])
        if role == "prompter" and lang == "en" and 40 <= len(t) <= 600
        and not re.search(r"\d", t)
    }
    if len(turns) < 1000:
        raise RuntimeError(f"only {len(turns)} usable filler turns")
    return sorted(turns)


def make_words(rng, filler, k, lead):
    """Word needle: an invented multi-token pet name plus a city, both asked for at the end."""
    owner, animal = rng.choice(OWNER), rng.choice(ANIMAL)
    name, city = f"{rng.choice(TITLE)} {rng.choice(PET)}", rng.choice(CITY)
    banned = [w.lower() for w in (*name.split(), city)]
    pool = [t for t in filler if not any(w in t.lower() for w in banned)]
    return {
        "needle_type": "words",
        "key": f"{owner}'s {animal}",
        "name": name,
        "city": city,
        "value": f"{name} | {city}",
        "needle": f"My {owner}'s {animal} is called {name} and lives in {city}.",
        "question": f"What is my {owner}'s {animal} called, and in which city does it live? "
                    "Answer with the name and the city only.",
        "fillers": rng.sample(pool, lead + k),
    }


def make_dialogue(rng, filler, k, idx, prefix="", lead=0, needle_type="digits"):
    if needle_type == "words":
        w = make_words(rng, filler, k, lead)
        fillers = w.pop("fillers")
        question = w.pop("question")
        row = {"id": f"{prefix}w_lead{lead}_k{k}_{idx:03d}", "k": k, "needle_turn": lead, **w, "system": SYSTEM}
        row["user_turns"] = [*fillers[:lead], row["needle"], *fillers[lead:], question]
        return row
    key = f"{rng.choice(ADJ)} {rng.choice(NOUN)} {rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randint(1, 9)}"
    value = str(rng.randint(10000, 99999))
    needle = f"Please remember this for later: the access code for the {key} is {value}."
    question = f"What is the access code for the {key}? Answer with the number only."
    if lead == 0:
        fillers = rng.sample(filler, k)
        return {
            "id": f"{prefix}k{k}_{idx:03d}",
            "k": k,
            "key": key,
            "value": value,
            "system": SYSTEM,
            "needle": needle,
            "user_turns": [needle, *fillers, question],
        }
    fillers = rng.sample(filler, lead + k)
    return {
        "id": f"{prefix}lead{lead}_k{k}_{idx:03d}",
        "k": k,
        "needle_turn": lead,
        "key": key,
        "value": value,
        "system": SYSTEM,
        "needle": needle,
        "user_turns": [*fillers[:lead], needle, *fillers[lead:], question],
    }


def build_rows(filler, seed, n_per_k=N_PER_K, prefix="", lead=0, needle_type="digits"):
    rng = random.Random(seed)
    return [make_dialogue(rng, filler, k, i, prefix, lead, needle_type) for k in K_VALUES for i in range(n_per_k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mt_niah.jsonl")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n_per_k", type=int, default=N_PER_K)
    ap.add_argument("--id_prefix", default="", help="e.g. s1_ so ids never collide with the seed-0 set")
    ap.add_argument("--lead", type=int, default=0,
                    help="filler turns BEFORE the needle (needle no longer at the start of the dialogue)")
    ap.add_argument("--needle", default="digits", choices=["digits", "words"],
                    help="digits = 5-digit access code (original); words = invented pet name + city")
    args = ap.parse_args()
    filler = load_filler()
    rows = build_rows(filler, args.seed, args.n_per_k, args.id_prefix, args.lead, args.needle)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} {args.needle} dialogues to {args.out} (seed {args.seed}, filler pool {len(filler)})")


if __name__ == "__main__":
    main()
