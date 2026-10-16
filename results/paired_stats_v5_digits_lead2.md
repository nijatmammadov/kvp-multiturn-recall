## Accuracy (95% Wilson CI)

| method | b | k | correct / n | acc | CI |
|---|---|---|---|---|---|
| full | 0 | 2 | 60 / 60 | 1.000 | [0.94, 1.00] |
| full | 0 | 4 | 60 / 60 | 1.000 | [0.94, 1.00] |
| full | 0 | 8 | 60 / 60 | 1.000 | [0.94, 1.00] |
| h2o | 256 | 2 | 3 / 60 | 0.050 | [0.02, 0.14] |
| h2o | 256 | 4 | 9 / 60 | 0.150 | [0.08, 0.26] |
| h2o | 256 | 8 | 2 / 60 | 0.033 | [0.01, 0.11] |
| kvp | 256 | 2 | 14 / 60 | 0.233 | [0.14, 0.35] |
| kvp | 256 | 4 | 3 / 60 | 0.050 | [0.02, 0.14] |
| kvp | 256 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| kvp_ft | 256 | 2 | 22 / 60 | 0.367 | [0.26, 0.49] |
| kvp_ft | 256 | 4 | 8 / 60 | 0.133 | [0.07, 0.24] |
| kvp_ft | 256 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| kvp_ft_pool4 | 256 | 2 | 27 / 60 | 0.450 | [0.33, 0.58] |
| kvp_ft_pool4 | 256 | 4 | 4 / 60 | 0.067 | [0.03, 0.16] |
| kvp_ft_pool4 | 256 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| kvp_ft_pool8 | 256 | 2 | 21 / 60 | 0.350 | [0.24, 0.48] |
| kvp_ft_pool8 | 256 | 4 | 2 / 60 | 0.033 | [0.01, 0.11] |
| kvp_ft_pool8 | 256 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| streaming | 256 | 2 | 19 / 60 | 0.317 | [0.21, 0.44] |
| streaming | 256 | 4 | 1 / 60 | 0.017 | [0.00, 0.09] |
| streaming | 256 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| streaming_first64 | 256 | 2 | 7 / 60 | 0.117 | [0.06, 0.22] |
| streaming_first64 | 256 | 4 | 0 / 60 | 0.000 | [0.00, 0.06] |
| streaming_first64 | 256 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |

## Paired exact McNemar (12 tests, Holm-adjusted)

| A vs B | b | k | n | only A right | only B right | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| kvp_ft_pool4 vs kvp_ft | 256 | 2 | 60 | 13 | 8 | 0.38 | 1 |
| kvp_ft_pool4 vs kvp_ft | 256 | 4 | 60 | 3 | 7 | 0.34 | 1 |
| kvp_ft_pool4 vs kvp_ft | 256 | 8 | 60 | 0 | 0 | 1 | 1 |
| kvp_ft_pool8 vs kvp_ft | 256 | 2 | 60 | 11 | 12 | 1 | 1 |
| kvp_ft_pool8 vs kvp_ft | 256 | 4 | 60 | 1 | 7 | 0.07 | 0.63 |
| kvp_ft_pool8 vs kvp_ft | 256 | 8 | 60 | 0 | 0 | 1 | 1 |
| kvp_ft vs kvp | 256 | 2 | 60 | 10 | 2 | 0.039 | 0.46 |
| kvp_ft vs kvp | 256 | 4 | 60 | 5 | 0 | 0.062 | 0.62 |
| kvp_ft vs kvp | 256 | 8 | 60 | 0 | 0 | 1 | 1 |
| kvp_ft vs streaming | 256 | 2 | 60 | 12 | 9 | 0.66 | 1 |
| kvp_ft vs streaming | 256 | 4 | 60 | 8 | 1 | 0.039 | 0.46 |
| kvp_ft vs streaming | 256 | 8 | 60 | 0 | 0 | 1 | 1 |

## Pre-registered pooled test (k in [4, 8], not adjusted)

| A vs B | b | n | only A right | only B right | p |
|---|---|---|---|---|---|
| kvp_ft_pool4 vs kvp_ft | 256 | 120 | 3 | 7 | 0.34 |
| kvp_ft_pool8 vs kvp_ft | 256 | 120 | 1 | 7 | 0.07 |
| kvp_ft vs kvp | 256 | 120 | 5 | 0 | 0.062 |
| kvp_ft vs streaming | 256 | 120 | 8 | 1 | 0.039 |
