Conditioned on full:0 correct: 112 / 120 dialogues kept.

## Accuracy (95% Wilson CI)

| method | b | k | correct / n | acc | CI |
|---|---|---|---|---|---|
| full | 0 | 4 | 58 / 58 | 1.000 | [0.94, 1.00] |
| full | 0 | 8 | 54 / 54 | 1.000 | [0.93, 1.00] |
| kvp_ft | 256 | 4 | 4 / 58 | 0.069 | [0.03, 0.16] |
| kvp_ft | 256 | 8 | 0 / 54 | 0.000 | [0.00, 0.07] |
| kvp_ft_heads48 | 256 | 4 | 31 / 58 | 0.534 | [0.41, 0.66] |
| kvp_ft_heads48 | 256 | 8 | 4 / 54 | 0.074 | [0.03, 0.18] |
| kvp_ft_turn_heads12 | 256 | 4 | 11 / 58 | 0.190 | [0.11, 0.31] |
| kvp_ft_turn_heads12 | 256 | 8 | 0 / 54 | 0.000 | [0.00, 0.07] |
| kvp_ft_turn_heads24 | 256 | 4 | 32 / 58 | 0.552 | [0.42, 0.67] |
| kvp_ft_turn_heads24 | 256 | 8 | 8 / 54 | 0.148 | [0.08, 0.27] |
| kvp_ft_turn_heads48 | 256 | 4 | 58 / 58 | 1.000 | [0.94, 1.00] |
| kvp_ft_turn_heads48 | 256 | 8 | 54 / 54 | 1.000 | [0.93, 1.00] |
| kvp_ft_turn_heads6 | 256 | 4 | 7 / 58 | 0.121 | [0.06, 0.23] |
| kvp_ft_turn_heads6 | 256 | 8 | 0 / 54 | 0.000 | [0.00, 0.07] |

## Paired exact McNemar (4 tests, Holm-adjusted)

| A vs B | b | k | n | only A right | only B right | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| kvp_ft_turn_heads48 vs kvp_ft_heads48 | 256 | 4 | 58 | 27 | 0 | 1.5e-08 | 1.5e-08 |
| kvp_ft_turn_heads48 vs kvp_ft_heads48 | 256 | 8 | 54 | 50 | 0 | 1.8e-15 | 3.6e-15 |
| kvp_ft_turn_heads48 vs kvp_ft | 256 | 4 | 58 | 54 | 0 | 1.1e-16 | 4.4e-16 |
| kvp_ft_turn_heads48 vs kvp_ft | 256 | 8 | 54 | 54 | 0 | 1.1e-16 | 4.4e-16 |

## Pre-registered pooled test (k in [4, 8], not adjusted)

| A vs B | b | n | only A right | only B right | p |
|---|---|---|---|---|---|
| kvp_ft_turn_heads48 vs kvp_ft_heads48 | 256 | 112 | 77 | 0 | 1.3e-23 |
| kvp_ft_turn_heads48 vs kvp_ft | 256 | 112 | 108 | 0 | 6.2e-33 |
