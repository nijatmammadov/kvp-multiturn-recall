Conditioned on full:0 correct: 175 / 180 dialogues kept.

## Accuracy (95% Wilson CI)

| method | b | k | correct / n | acc | CI |
|---|---|---|---|---|---|
| full | 0 | 2 | 56 / 56 | 1.000 | [0.94, 1.00] |
| full | 0 | 4 | 59 / 59 | 1.000 | [0.94, 1.00] |
| full | 0 | 8 | 60 / 60 | 1.000 | [0.94, 1.00] |
| h2o | 256 | 2 | 40 / 56 | 0.714 | [0.59, 0.82] |
| h2o | 256 | 4 | 37 / 59 | 0.627 | [0.50, 0.74] |
| h2o | 256 | 8 | 23 / 60 | 0.383 | [0.27, 0.51] |
| kvp | 256 | 2 | 17 / 56 | 0.304 | [0.20, 0.43] |
| kvp | 256 | 4 | 0 / 59 | 0.000 | [0.00, 0.06] |
| kvp | 256 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| kvp_ft | 128 | 2 | 1 / 56 | 0.018 | [0.00, 0.09] |
| kvp_ft | 128 | 4 | 0 / 59 | 0.000 | [0.00, 0.06] |
| kvp_ft | 128 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| kvp_ft | 256 | 2 | 15 / 56 | 0.268 | [0.17, 0.40] |
| kvp_ft | 256 | 4 | 2 / 59 | 0.034 | [0.01, 0.12] |
| kvp_ft | 256 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| kvp_ft_pool4 | 128 | 2 | 0 / 56 | 0.000 | [0.00, 0.06] |
| kvp_ft_pool4 | 128 | 4 | 0 / 59 | 0.000 | [0.00, 0.06] |
| kvp_ft_pool4 | 128 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| kvp_ft_pool4 | 256 | 2 | 9 / 56 | 0.161 | [0.09, 0.28] |
| kvp_ft_pool4 | 256 | 4 | 0 / 59 | 0.000 | [0.00, 0.06] |
| kvp_ft_pool4 | 256 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| kvp_ft_pool8 | 128 | 2 | 1 / 56 | 0.018 | [0.00, 0.09] |
| kvp_ft_pool8 | 128 | 4 | 0 / 59 | 0.000 | [0.00, 0.06] |
| kvp_ft_pool8 | 128 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| kvp_ft_pool8 | 256 | 2 | 10 / 56 | 0.179 | [0.10, 0.30] |
| kvp_ft_pool8 | 256 | 4 | 0 / 59 | 0.000 | [0.00, 0.06] |
| kvp_ft_pool8 | 256 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| streaming | 256 | 2 | 12 / 56 | 0.214 | [0.13, 0.34] |
| streaming | 256 | 4 | 0 / 59 | 0.000 | [0.00, 0.06] |
| streaming | 256 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |

## Paired exact McNemar (21 tests, Holm-adjusted)

| A vs B | b | k | n | only A right | only B right | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| kvp_ft_pool4 vs kvp_ft | 128 | 2 | 56 | 0 | 1 | 1 | 1 |
| kvp_ft_pool4 vs kvp_ft | 128 | 4 | 59 | 0 | 0 | 1 | 1 |
| kvp_ft_pool4 vs kvp_ft | 128 | 8 | 60 | 0 | 0 | 1 | 1 |
| kvp_ft_pool4 vs kvp_ft | 256 | 2 | 56 | 4 | 10 | 0.18 | 1 |
| kvp_ft_pool4 vs kvp_ft | 256 | 4 | 59 | 0 | 2 | 0.5 | 1 |
| kvp_ft_pool4 vs kvp_ft | 256 | 8 | 60 | 0 | 0 | 1 | 1 |
| kvp_ft_pool8 vs kvp_ft | 128 | 2 | 56 | 1 | 1 | 1 | 1 |
| kvp_ft_pool8 vs kvp_ft | 128 | 4 | 59 | 0 | 0 | 1 | 1 |
| kvp_ft_pool8 vs kvp_ft | 128 | 8 | 60 | 0 | 0 | 1 | 1 |
| kvp_ft_pool8 vs kvp_ft | 256 | 2 | 56 | 4 | 9 | 0.27 | 1 |
| kvp_ft_pool8 vs kvp_ft | 256 | 4 | 59 | 0 | 2 | 0.5 | 1 |
| kvp_ft_pool8 vs kvp_ft | 256 | 8 | 60 | 0 | 0 | 1 | 1 |
| kvp_ft vs kvp | 256 | 2 | 56 | 2 | 4 | 0.69 | 1 |
| kvp_ft vs kvp | 256 | 4 | 59 | 2 | 0 | 0.5 | 1 |
| kvp_ft vs kvp | 256 | 8 | 60 | 0 | 0 | 1 | 1 |
| kvp_ft vs h2o | 256 | 2 | 56 | 0 | 25 | 6e-08 | 1.2e-06 |
| kvp_ft vs h2o | 256 | 4 | 59 | 0 | 35 | 5.8e-11 | 1.2e-09 |
| kvp_ft vs h2o | 256 | 8 | 60 | 0 | 23 | 2.4e-07 | 4.5e-06 |
| kvp_ft vs streaming | 256 | 2 | 56 | 9 | 6 | 0.61 | 1 |
| kvp_ft vs streaming | 256 | 4 | 59 | 2 | 0 | 0.5 | 1 |
| kvp_ft vs streaming | 256 | 8 | 60 | 0 | 0 | 1 | 1 |

## Pre-registered pooled test (k in [4, 8], not adjusted)

| A vs B | b | n | only A right | only B right | p |
|---|---|---|---|---|---|
| kvp_ft_pool4 vs kvp_ft | 128 | 119 | 0 | 0 | 1 |
| kvp_ft_pool4 vs kvp_ft | 256 | 119 | 0 | 2 | 0.5 |
| kvp_ft_pool8 vs kvp_ft | 128 | 119 | 0 | 0 | 1 |
| kvp_ft_pool8 vs kvp_ft | 256 | 119 | 0 | 2 | 0.5 |
| kvp_ft vs kvp | 256 | 119 | 2 | 0 | 0.5 |
| kvp_ft vs h2o | 256 | 119 | 0 | 58 | 6.9e-18 |
| kvp_ft vs streaming | 256 | 119 | 2 | 0 | 0.5 |
