# Is the mechanism a small-model effect? (batch v8)

Words q1, StreamingLLM base, b = 256 per head, k in {4, 8}, fp32 (Kaggle T4); protection in all heads. Each model conditioned on its own full-cache run.

| model | full cache (all k = 4, 8) | config | correct / n | acc | 95% CI |
|---|---|---|---|---|---|
| Qwen/Qwen2.5-1.5B-Instruct | 117 / 120 | streaming | 0 / 117 | 0.000 | [0.00, 0.03] |
| Qwen/Qwen2.5-1.5B-Instruct | 117 / 120 | value | 0 / 117 | 0.000 | [0.00, 0.03] |
| Qwen/Qwen2.5-1.5B-Instruct | 117 / 120 | keyval | 116 / 117 | 0.991 | [0.95, 1.00] |
| Qwen/Qwen2.5-1.5B-Instruct | 117 / 120 | turn | 116 / 117 | 0.991 | [0.95, 1.00] |
| Qwen/Qwen2.5-1.5B-Instruct | 117 / 120 | offload | 116 / 117 | 0.991 | [0.95, 1.00] |
| Qwen/Qwen2.5-0.5B-Instruct | 112 / 120 | streaming | 0 / 112 | 0.000 | [0.00, 0.03] |
| Qwen/Qwen2.5-0.5B-Instruct | 112 / 120 | value | 0 / 112 | 0.000 | [0.00, 0.03] |
| Qwen/Qwen2.5-0.5B-Instruct | 112 / 120 | keyval | 105 / 112 | 0.938 | [0.88, 0.97] |
| Qwen/Qwen2.5-0.5B-Instruct | 112 / 120 | turn | 112 / 112 | 1.000 | [0.97, 1.00] |
| Qwen/Qwen2.5-0.5B-Instruct | 112 / 120 | offload | 111 / 112 | 0.991 | [0.95, 1.00] |

**M0 (read first, Qwen/Qwen2.5-1.5B-Instruct): full cache 0.975 on k = 4, 8 (needs >= 0.80) -> MET.**
**M1 (Qwen/Qwen2.5-1.5B-Instruct): turn vs value: only first right 116, only second right 0, p = 2.4e-35, p_holm = 7.2e-35 -> MET.**
**M2 (Qwen/Qwen2.5-1.5B-Instruct): keyval vs value: only first right 116, only second right 0, p = 2.4e-35, p_holm = 7.2e-35 -> MET.**
**M3 (Qwen/Qwen2.5-1.5B-Instruct): offload vs streaming: only first right 116, only second right 0, p = 2.4e-35, p_holm = 7.2e-35 -> MET.**
Offload retrieval hit at the question (Qwen/Qwen2.5-1.5B-Instruct): 117 / 117.

Descriptive, same tests on Qwen/Qwen2.5-0.5B-Instruct (not adjusted):
- turn vs value: only first right 112, only second right 0, p = 3.9e-34
- keyval vs value: only first right 105, only second right 0, p = 4.9e-32
- offload vs streaming: only first right 111, only second right 0, p = 7.7e-34
Cross-size (value, 109 dialogues both models' full cache got right): only Qwen/Qwen2.5-1.5B-Instruct right 0, only Qwen/Qwen2.5-0.5B-Instruct right 0, p = 1.
Cross-size (keyval, 109 dialogues both models' full cache got right): only Qwen/Qwen2.5-1.5B-Instruct right 7, only Qwen/Qwen2.5-0.5B-Instruct right 1, p = 0.07.
Cross-size (streaming, 109 dialogues both models' full cache got right): only Qwen/Qwen2.5-1.5B-Instruct right 0, only Qwen/Qwen2.5-0.5B-Instruct right 0, p = 1.

Reading: the mechanism holds at the larger size if M1-M3 are met there. If value alone reaches the turn accuracy at 1.5B, the need for the linking context was a small-model effect.
