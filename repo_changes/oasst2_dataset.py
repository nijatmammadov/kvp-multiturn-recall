import random
from collections import defaultdict

import torch
from datasets import load_dataset
from torchtune.data import Message

SYSTEM = "You are a helpful assistant."
ROLE = {"prompter": "user", "assistant": "assistant"}


def _best_child(children):
    ranked = [c for c in children if c["rank"] is not None]
    return min(ranked, key=lambda c: c["rank"]) if ranked else children[0]


def _threads(rows):
    by_parent = defaultdict(list)
    roots = []
    for r in rows:
        if r["parent_id"] is None:
            roots.append(r)
        else:
            by_parent[r["parent_id"]].append(r)
    for root in roots:
        path = [root]
        while by_parent.get(path[-1]["message_id"]):
            path.append(_best_child(by_parent[path[-1]["message_id"]]))
        yield path


class OASST2MultiTurnDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, n_samples=500, min_prompt_tokens=200,
                 max_prompt_tokens=1200, min_user_turns=2, seed=0):
        super().__init__()
        self.tokenizer = tokenizer
        ds = load_dataset("OpenAssistant/oasst2", split="train")
        rows = [r for r in ds if r["lang"] == "en" and not r["deleted"]]

        candidates = []
        for path in _threads(rows):
            if path[-1]["role"] != "assistant":
                path = path[:-1]
            prompt_path = path[:-1]
            if not prompt_path or prompt_path[-1]["role"] != "prompter":
                continue
            if sum(m["role"] == "prompter" for m in prompt_path) < min_user_turns:
                continue
            roles = [m["role"] for m in prompt_path]
            if any(a == b for a, b in zip(roles, roles[1:])):
                raise ValueError(f"non-alternating thread {prompt_path[0]['message_tree_id']}")
            tokens, mask = self._tokenize(prompt_path)
            if min_prompt_tokens <= len(tokens) <= max_prompt_tokens:
                candidates.append((prompt_path[0]["message_tree_id"], tokens, mask))

        candidates.sort(key=lambda c: c[0])
        random.Random(seed).shuffle(candidates)
        if len(candidates) < n_samples:
            raise RuntimeError(f"only {len(candidates)} threads pass the filters, need {n_samples}")
        self.samples = candidates[:n_samples]

    def _tokenize(self, prompt_path):
        messages = [Message(role="system", content=SYSTEM)]
        messages += [Message(role=ROLE[m["role"]], content=m["text"]) for m in prompt_path]
        messages.append(Message(role="assistant", content="", eot=False))
        return self.tokenizer.tokenize_messages(messages, add_eos=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, tokens, mask = self.samples[idx]
        return {"tokens": tokens, "mask": mask, "id": idx}
