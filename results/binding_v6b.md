# Which part of the needle turn must survive (batch v6, amendment 3)

kvp_ft, b = 256, protection in all 48 heads, k in {4, 8}, fp32 (Kaggle T4), conditioned on full-cache correct.

| set | protected | correct / n | acc | 95% CI | mean forced tokens |
|---|---|---|---|---|---|
| words | value | 35 / 112 | 0.312 | [0.23, 0.40] | n/a (older run) |
| words | keyval | 88 / 112 | 0.786 | [0.70, 0.85] | 8.9 |
| words | turn | 112 / 112 | 1.000 | [0.97, 1.00] | n/a (older run) |
| digits | value | 111 / 120 | 0.925 | [0.86, 0.96] | 5.0 |
| digits | keyval | 119 / 120 | 0.992 | [0.95, 1.00] | 9.2 |
| digits | turn | 120 / 120 | 1.000 | [0.97, 1.00] | 32.2 |

**D0 (manipulation check, read first): digits turn accuracy = 1.000 (needs >= 0.80) -> MET.**
**K1 (primary, words): keyval vs value: only keyval right 55, only value right 2, p = 2.3e-14, p_holm = 4.6e-14 -> MET.**
**D1 (primary, digits): turn vs value: only turn right 9, only value right 0, p = 0.0039, p_holm = 0.0039 -> MET.**

K2 (descriptive, words): keyval 0.786 vs turn 1.000 (ratio 0.79); paired: only keyval right 0, only turn right 24, p = 1.2e-07. Ratio >= 0.90: the linking phrase is enough; lower: other words of the turn are needed too.
D2 (descriptive, digits): keyval vs value: only keyval right 9, only value right 1, p = 0.021.
D2 (descriptive, digits): keyval vs turn: only keyval right 0, only turn right 1, p = 1.
