"""CPU test for chat_evict.py +fix: tiny random Qwen2, a toy word tokenizer, no downloads. Run from eval/."""
import torch

from chat_evict import Conversation
from test_v6 import tiny

BUDGET = 120


class ToyTok:
    """One token per character (ids 1..127); chat specials are single ids."""
    SPECIAL = {"<|im_start|>": 1, "<|im_end|>": 2}

    def convert_tokens_to_ids(self, t):
        return self.SPECIAL[t]

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            for sp, v in self.SPECIAL.items():
                if text.startswith(sp, i):
                    ids.append(v)
                    i += len(sp)
                    break
            else:
                ids.append(3 + ord(text[i]) % 120)
                i += 1
        return type("Enc", (), {"input_ids": ids})()

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + t % 26) for t in ids if not (skip_special_tokens and t in (1, 2)))


def test_fix_restores_matching_message():
    model, agents = tiny()
    tok = ToyTok()
    fillers = ["tell me about rivers and lakes in the north", "how do trains run on time every day",
               "what about clouds and rain today", "explain how bread is baked at home", "why is the sky blue"]
    msgs = ["my hamster lives in bergen"] + fillers * 3 + ["where does my hamster live"]
    pick = lambda past, q: max(range(len(past)), key=lambda j: len(set(past[j].split()) & set(q.split())))
    with torch.no_grad():
        for method in ("kvp+fix", "streaming+fix"):
            c = Conversation(method, model, tok, {2}, 2, BUDGET, agents, 0, picker=pick)
            for m in msgs:
                c.ask(m, 6, show=False)
                assert c.state.cache_len() <= BUDGET
            assert c.state.next_pos > 3 * BUDGET, "test is vacuous: little was evicted"
            assert c.note.startswith("restored your message #1"), c.note
            R = c.user_ranges[0]
            assert all(set(R) <= set(row) for p in c.state.pos for row in p.tolist()), "message 1 not in all heads"
            assert len(c.stream) == c.state.next_pos
            want = tok("<|im_start|>user\nmy hamster lives in bergen<|im_end|>\n<|im_start|>assistant\n").input_ids
            assert c.stream[R[0]:R[-1] + 1] == want, "restored range is not exactly user message #1"
    plain = Conversation("kvp", model, tok, {2}, 2, BUDGET, agents, 0)
    with torch.no_grad():
        for m in msgs:
            plain.ask(m, 6, show=False)
    assert plain.note == "" and not plain.state.offload


def test_fix_suffix_checks():
    model, agents = tiny()
    for bad in ("h2o+fix", "full+fix", "kvp+fux"):
        try:
            Conversation(bad, model, ToyTok(), {2}, 2, BUDGET, agents, 0)
        except SystemExit:
            continue
        raise AssertionError(f"{bad} accepted")


if __name__ == "__main__":
    test_fix_restores_matching_message()
    print("PASS test_fix_restores_matching_message")
    test_fix_suffix_checks()
    print("PASS test_fix_suffix_checks")
