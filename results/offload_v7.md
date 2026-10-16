# Offload + retrieve (batch v7)

b = 256, k in {4, 8}, fp32 (Kaggle T4), conditioned on each set's full-cache run answering correctly.

| set | config | correct / n | acc | 95% CI |
|---|---|---|---|---|
| words | kvp_ft | 4 / 112 | 0.036 | [0.01, 0.09] |
| words | offload | 112 / 112 | 1.000 | [0.97, 1.00] |
| words | h2o | 59 / 112 | 0.527 | [0.43, 0.62] |
| words | oracle | 112 / 112 | 1.000 | [0.97, 1.00] |
| digits | kvp_ft | 5 / 120 | 0.042 | [0.02, 0.09] |
| digits | offload | 120 / 120 | 1.000 | [0.97, 1.00] |
| digits | h2o | 15 / 120 | 0.125 | [0.08, 0.20] |
| digits | oracle | 120 / 120 | 1.000 | [0.97, 1.00] |

R0 (words): retrieval picked the needle turn at the question in 112 / 112; accuracy when hit 112 / 112, when missed 0 / 0; restored tokens at the question: mean 24.9.
R0 (digits): retrieval picked the needle turn at the question in 120 / 120; accuracy when hit 120 / 120, when missed 0 / 0; restored tokens at the question: mean 32.2.

**F1 (primary, words): offload vs kvp_ft: only offload right 108, only kvp_ft right 0, p = 6.2e-33, p_holm = 6.2e-33 -> MET.**
**F2 (primary, digits): offload vs kvp_ft: only offload right 115, only kvp_ft right 0, p = 4.8e-35, p_holm = 9.6e-35 -> MET.**
S1 (secondary, words): offload vs h2o: only offload right 53, only h2o right 0, p = 2.2e-16, p_holm = 2.2e-16 -> MET (offload better).
S2 (secondary, digits): offload vs h2o: only offload right 105, only h2o right 0, p = 4.9e-32, p_holm = 9.9e-32 -> MET (offload better).
Descriptive (words): oracle (turn kept in all heads) vs offload: only oracle right 0, only offload right 0, p = 1.
Descriptive (digits): oracle (turn kept in all heads) vs offload: only oracle right 0, only offload right 0, p = 1.

Caveat (pre-registered): retrieval is lexical and the benchmark questions repeat the needle's words, so R0 is near-perfect by construction; this measures the eviction side, not retrieval under paraphrase.
