# Head-breadth test (batch v6, rules B1-B2)

Words set (q1), b = 256, kvp_ft agents, k in [4, 8], conditioned on full-cache correct: 112 dialogues per level. Force-kept (whole needle turn) in n of 48 (layer, KV head) pairs, configs kvp_ft_turn_heads{n}; n = 0 is results/logs_eval_v6k_fp32/kvp_ft_b256_fp32.jsonl (no protection).

| n heads | correct / n | acc | 95% CI | name_ok | city_ok | heads whole (mean) | needle survival (mean) |
|---|---|---|---|---|---|---|---|
| 0 | 4 / 112 | 0.036 | [0.01, 0.09] | 0.054 | 0.125 | 0.020 | 0.176 |
| 6 | 7 / 112 | 0.062 | [0.03, 0.12] | 0.089 | 0.179 | 0.142 | 0.279 |
| 12 | 11 / 112 | 0.098 | [0.06, 0.17] | 0.134 | 0.241 | 0.264 | 0.381 |
| 24 | 40 / 112 | 0.357 | [0.27, 0.45] | 0.402 | 0.634 | 0.509 | 0.586 |
| 48 | 112 / 112 | 1.000 | [0.97, 1.00] | 1.000 | 1.000 | 1.000 | 1.000 |

**B2 (manipulation check, read first): accuracy at n = 48 = 1.000 (needs >= 0.80) -> MET.**
**B1 (primary): Cochran-Armitage trend over n = [0, 6, 12, 24, 48]: z = 18.18, one-sided p = 4e-74 -> MET.**

Descriptive: n = 48 vs n = 0, exact McNemar: only n=48 right 108, only n=0 right 0, p = 6.2e-33.
