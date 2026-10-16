import csv

CATS = ["template", "system", "needle_turn", "filler_user", "question", "assistant"]
print("| method | b | k | " + " | ".join(CATS) + " |")
print("|---|---|---|" + "---|" * len(CATS))
for r in csv.DictReader(open("analysis/summary.csv")):
    cells = [f"{float(r['kept_' + c]):.3f} / {float(r['base_' + c]):.3f}" for c in CATS]
    print(f"| {r['method']} | {r['budget']} | {r['k']} | " + " | ".join(cells) + " |")
print("\ncells = kept share / base share (fraction of tokens seen so far)")
