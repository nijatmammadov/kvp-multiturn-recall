import sys

from torchtune.data import Message
from torchtune.models.qwen2_5 import qwen2_5_tokenizer
from transformers import AutoTokenizer

model_dir = sys.argv[1]
convo = [
    ("system", "You are a helpful assistant."),
    ("user", "Remember this: the access code for vault Q7 is 84920."),
    ("assistant", "Got it."),
    ("user", "Give me three tips for growing tomatoes."),
]

tt = qwen2_5_tokenizer(path=f"{model_dir}/vocab.json", merges_file=f"{model_dir}/merges.txt", max_seq_len=None)
msgs = [Message(role=r, content=c) for r, c in convo]
msgs.append(Message(role="assistant", content="", eot=False))
tt_tokens, _ = tt.tokenize_messages(msgs, add_eos=False)

hf = AutoTokenizer.from_pretrained(model_dir)
hf_text = hf.apply_chat_template([{"role": r, "content": c} for r, c in convo],
                                 tokenize=False, add_generation_prompt=True)
hf_tokens = hf(hf_text, add_special_tokens=False).input_ids

print("torchtune:", repr(tt.decode(tt_tokens, skip_special_tokens=False)))
print("hf       :", repr(hf_text))
if tt_tokens != hf_tokens:
    raise SystemExit(f"MISMATCH: torchtune {len(tt_tokens)} tokens vs hf {len(hf_tokens)} tokens")
print(f"MATCH: {len(tt_tokens)} tokens")
