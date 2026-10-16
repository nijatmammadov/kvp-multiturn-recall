Conditioned on full:0 correct: 120 / 120 dialogues kept.

## Accuracy (95% Wilson CI)

| method | b | k | correct / n | acc | CI |
|---|---|---|---|---|---|
| full | 0 | 4 | 60 / 60 | 1.000 | [0.94, 1.00] |
| full | 0 | 8 | 60 / 60 | 1.000 | [0.94, 1.00] |
| kvp_ft_heads48 | 256 | 4 | 59 / 60 | 0.983 | [0.91, 1.00] |
| kvp_ft_heads48 | 256 | 8 | 52 / 60 | 0.867 | [0.76, 0.93] |
| kvp_ft_keyval_heads48 | 256 | 4 | 59 / 60 | 0.983 | [0.91, 1.00] |
| kvp_ft_keyval_heads48 | 256 | 8 | 60 / 60 | 1.000 | [0.94, 1.00] |
| kvp_ft_turn_heads48 | 256 | 4 | 60 / 60 | 1.000 | [0.94, 1.00] |
| kvp_ft_turn_heads48 | 256 | 8 | 60 / 60 | 1.000 | [0.94, 1.00] |

## Paired exact McNemar (6 tests, Holm-adjusted)

| A vs B | b | k | n | only A right | only B right | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| kvp_ft_turn_heads48 vs kvp_ft_heads48 | 256 | 4 | 60 | 1 | 0 | 1 | 1 |
| kvp_ft_turn_heads48 vs kvp_ft_heads48 | 256 | 8 | 60 | 8 | 0 | 0.0078 | 0.047 |
| kvp_ft_keyval_heads48 vs kvp_ft_heads48 | 256 | 4 | 60 | 1 | 1 | 1 | 1 |
| kvp_ft_keyval_heads48 vs kvp_ft_heads48 | 256 | 8 | 60 | 8 | 0 | 0.0078 | 0.047 |
| kvp_ft_turn_heads48 vs kvp_ft_keyval_heads48 | 256 | 4 | 60 | 1 | 0 | 1 | 1 |
| kvp_ft_turn_heads48 vs kvp_ft_keyval_heads48 | 256 | 8 | 60 | 0 | 0 | 1 | 1 |

## Pre-registered pooled test (k in [4, 8], not adjusted)

| A vs B | b | n | only A right | only B right | p |
|---|---|---|---|---|---|
| kvp_ft_turn_heads48 vs kvp_ft_heads48 | 256 | 120 | 9 | 0 | 0.0039 |
| kvp_ft_keyval_heads48 vs kvp_ft_heads48 | 256 | 120 | 9 | 1 | 0.021 |
| kvp_ft_turn_heads48 vs kvp_ft_keyval_heads48 | 256 | 120 | 1 | 0 | 1 |
