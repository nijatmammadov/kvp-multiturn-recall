# data/

- `recall_prompts.jsonl`: 600 short RULER-style recall prompts for the batch v4 fine-tune, built by
  `python eval/make_recall_prompts.py --wikitext <wikitext-2/train.txt> --out data/recall_prompts.jsonl` (seed 0).
  Haystack text: WikiText-2 (Merity et al., 2016), CC BY-SA 3.0, tokenized release as shipped in
  pytorch/examples (`word_language_model/data/wikitext-2/train.txt`); sentences with `<unk>` dropped, spacing
  detokenized. Fields: `id`, `variant` (single / multikey / multivalue / multiquery), `vtype` (numbers / uuids /
  words), `system`, `user`, `answer_prefix`, `answers`, `asked`, `decoys`.
