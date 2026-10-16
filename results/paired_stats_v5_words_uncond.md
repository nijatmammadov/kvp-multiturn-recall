## Accuracy (95% Wilson CI)

| method | b | k | correct / n | acc | CI |
|---|---|---|---|---|---|
| full | 0 | 2 | 56 / 60 | 0.933 | [0.84, 0.97] |
| full | 0 | 4 | 59 / 60 | 0.983 | [0.91, 1.00] |
| full | 0 | 8 | 60 / 60 | 1.000 | [0.94, 1.00] |
| h2o | 256 | 2 | 42 / 60 | 0.700 | [0.57, 0.80] |
| h2o | 256 | 4 | 37 / 60 | 0.617 | [0.49, 0.73] |
| h2o | 256 | 8 | 23 / 60 | 0.383 | [0.27, 0.51] |
| kvp | 256 | 2 | 17 / 60 | 0.283 | [0.19, 0.41] |
| kvp | 256 | 4 | 0 / 60 | 0.000 | [0.00, 0.06] |
| kvp | 256 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| kvp_ft | 128 | 2 | 1 / 60 | 0.017 | [0.00, 0.09] |
| kvp_ft | 128 | 4 | 0 / 60 | 0.000 | [0.00, 0.06] |
| kvp_ft | 128 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| kvp_ft | 256 | 2 | 15 / 60 | 0.250 | [0.16, 0.37] |
| kvp_ft | 256 | 4 | 2 / 60 | 0.033 | [0.01, 0.11] |
| kvp_ft | 256 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| kvp_ft_pool4 | 128 | 2 | 0 / 60 | 0.000 | [0.00, 0.06] |
| kvp_ft_pool4 | 128 | 4 | 0 / 60 | 0.000 | [0.00, 0.06] |
| kvp_ft_pool4 | 128 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| kvp_ft_pool4 | 256 | 2 | 9 / 60 | 0.150 | [0.08, 0.26] |
| kvp_ft_pool4 | 256 | 4 | 0 / 60 | 0.000 | [0.00, 0.06] |
| kvp_ft_pool4 | 256 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| kvp_ft_pool8 | 128 | 2 | 1 / 60 | 0.017 | [0.00, 0.09] |
| kvp_ft_pool8 | 128 | 4 | 0 / 60 | 0.000 | [0.00, 0.06] |
| kvp_ft_pool8 | 128 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| kvp_ft_pool8 | 256 | 2 | 10 / 60 | 0.167 | [0.09, 0.28] |
| kvp_ft_pool8 | 256 | 4 | 0 / 60 | 0.000 | [0.00, 0.06] |
| kvp_ft_pool8 | 256 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| streaming | 256 | 2 | 13 / 60 | 0.217 | [0.13, 0.34] |
| streaming | 256 | 4 | 0 / 60 | 0.000 | [0.00, 0.06] |
| streaming | 256 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |

## Paired exact McNemar (21 tests, Holm-adjusted)

| A vs B | b | k | n | only A right | only B right | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| kvp_ft_pool4 vs kvp_ft | 128 | 2 | 60 | 0 | 1 | 1 | 1 |
| kvp_ft_pool4 vs kvp_ft | 128 | 4 | 60 | 0 | 0 | 1 | 1 |
| kvp_ft_pool4 vs kvp_ft | 128 | 8 | 60 | 0 | 0 | 1 | 1 |
| kvp_ft_pool4 vs kvp_ft | 256 | 2 | 60 | 4 | 10 | 0.18 | 1 |
| kvp_ft_pool4 vs kvp_ft | 256 | 4 | 60 | 0 | 2 | 0.5 | 1 |
| kvp_ft_pool4 vs kvp_ft | 256 | 8 | 60 | 0 | 0 | 1 | 1 |
| kvp_ft_pool8 vs kvp_ft | 128 | 2 | 60 | 1 | 1 | 1 | 1 |
| kvp_ft_pool8 vs kvp_ft | 128 | 4 | 60 | 0 | 0 | 1 | 1 |
| kvp_ft_pool8 vs kvp_ft | 128 | 8 | 60 | 0 | 0 | 1 | 1 |
| kvp_ft_pool8 vs kvp_ft | 256 | 2 | 60 | 4 | 9 | 0.27 | 1 |
| kvp_ft_pool8 vs kvp_ft | 256 | 4 | 60 | 0 | 2 | 0.5 | 1 |
| kvp_ft_pool8 vs kvp_ft | 256 | 8 | 60 | 0 | 0 | 1 | 1 |
| kvp_ft vs kvp | 256 | 2 | 60 | 2 | 4 | 0.69 | 1 |
| kvp_ft vs kvp | 256 | 4 | 60 | 2 | 0 | 0.5 | 1 |
| kvp_ft vs kvp | 256 | 8 | 60 | 0 | 0 | 1 | 1 |
| kvp_ft vs h2o | 256 | 2 | 60 | 0 | 27 | 1.5e-08 | 3e-07 |
| kvp_ft vs h2o | 256 | 4 | 60 | 0 | 35 | 5.8e-11 | 1.2e-09 |
| kvp_ft vs h2o | 256 | 8 | 60 | 0 | 23 | 2.4e-07 | 4.5e-06 |
| kvp_ft vs streaming | 256 | 2 | 60 | 9 | 7 | 0.8 | 1 |
| kvp_ft vs streaming | 256 | 4 | 60 | 2 | 0 | 0.5 | 1 |
| kvp_ft vs streaming | 256 | 8 | 60 | 0 | 0 | 1 | 1 |

## Pre-registered pooled test (k in [4, 8], not adjusted)

| A vs B | b | n | only A right | only B right | p |
|---|---|---|---|---|---|
| kvp_ft_pool4 vs kvp_ft | 128 | 120 | 0 | 0 | 1 |
| kvp_ft_pool4 vs kvp_ft | 256 | 120 | 0 | 2 | 0.5 |
| kvp_ft_pool8 vs kvp_ft | 128 | 120 | 0 | 0 | 1 |
| kvp_ft_pool8 vs kvp_ft | 256 | 120 | 0 | 2 | 0.5 |
| kvp_ft vs kvp | 256 | 120 | 2 | 0 | 0.5 |
| kvp_ft vs h2o | 256 | 120 | 0 | 58 | 6.9e-18 |
| kvp_ft vs streaming | 256 | 120 | 2 | 0 | 0.5 |
