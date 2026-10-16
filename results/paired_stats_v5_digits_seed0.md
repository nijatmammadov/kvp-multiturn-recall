## Accuracy (95% Wilson CI)

| method | b | k | correct / n | acc | CI |
|---|---|---|---|---|---|
| kvp_ft | 128 | 2 | 22 / 60 | 0.367 | [0.26, 0.49] |
| kvp_ft | 128 | 4 | 2 / 60 | 0.033 | [0.01, 0.11] |
| kvp_ft | 128 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| kvp_ft | 256 | 2 | 57 / 60 | 0.950 | [0.86, 0.98] |
| kvp_ft | 256 | 4 | 27 / 60 | 0.450 | [0.33, 0.58] |
| kvp_ft | 256 | 8 | 2 / 60 | 0.033 | [0.01, 0.11] |
| kvp_ft_pool4 | 256 | 2 | 59 / 60 | 0.983 | [0.91, 1.00] |
| kvp_ft_pool4 | 256 | 4 | 13 / 60 | 0.217 | [0.13, 0.34] |
| kvp_ft_pool4 | 256 | 8 | 2 / 60 | 0.033 | [0.01, 0.11] |
| kvp_ft_pool8 | 256 | 2 | 60 / 60 | 1.000 | [0.94, 1.00] |
| kvp_ft_pool8 | 256 | 4 | 33 / 60 | 0.550 | [0.42, 0.67] |
| kvp_ft_pool8 | 256 | 8 | 1 / 60 | 0.017 | [0.00, 0.09] |
| kvp_rerun | 256 | 2 | 52 / 60 | 0.867 | [0.76, 0.93] |
| kvp_rerun | 256 | 4 | 20 / 60 | 0.333 | [0.23, 0.46] |
| kvp_rerun | 256 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| streaming_first64 | 128 | 2 | 60 / 60 | 1.000 | [0.94, 1.00] |
| streaming_first64 | 128 | 4 | 60 / 60 | 1.000 | [0.94, 1.00] |
| streaming_first64 | 128 | 8 | 60 / 60 | 1.000 | [0.94, 1.00] |
| streaming_first64 | 256 | 2 | 60 / 60 | 1.000 | [0.94, 1.00] |
| streaming_first64 | 256 | 4 | 60 / 60 | 1.000 | [0.94, 1.00] |
| streaming_first64 | 256 | 8 | 60 / 60 | 1.000 | [0.94, 1.00] |

## Paired exact McNemar (6 tests, Holm-adjusted)

| A vs B | b | k | n | only A right | only B right | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| kvp_ft_pool4 vs kvp_ft | 256 | 2 | 60 | 3 | 1 | 0.62 | 1 |
| kvp_ft_pool4 vs kvp_ft | 256 | 4 | 60 | 3 | 17 | 0.0026 | 0.015 |
| kvp_ft_pool4 vs kvp_ft | 256 | 8 | 60 | 1 | 1 | 1 | 1 |
| kvp_ft_pool8 vs kvp_ft | 256 | 2 | 60 | 3 | 0 | 0.25 | 1 |
| kvp_ft_pool8 vs kvp_ft | 256 | 4 | 60 | 13 | 7 | 0.26 | 1 |
| kvp_ft_pool8 vs kvp_ft | 256 | 8 | 60 | 1 | 2 | 1 | 1 |

## Pre-registered pooled test (k in [4, 8], not adjusted)

| A vs B | b | n | only A right | only B right | p |
|---|---|---|---|---|---|
| kvp_ft_pool4 vs kvp_ft | 256 | 120 | 4 | 18 | 0.0043 |
| kvp_ft_pool8 vs kvp_ft | 256 | 120 | 14 | 9 | 0.4 |
