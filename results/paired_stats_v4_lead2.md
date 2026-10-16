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
| streaming | 256 | 2 | 19 / 60 | 0.317 | [0.21, 0.44] |
| streaming | 256 | 4 | 1 / 60 | 0.017 | [0.00, 0.09] |
| streaming | 256 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| streaming_first64 | 256 | 2 | 7 / 60 | 0.117 | [0.06, 0.22] |
| streaming_first64 | 256 | 4 | 0 / 60 | 0.000 | [0.00, 0.06] |
| streaming_first64 | 256 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |

## Paired exact McNemar (9 tests, Holm-adjusted)

| A vs B | b | k | n | only A right | only B right | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| kvp vs h2o | 256 | 2 | 60 | 12 | 1 | 0.0034 | 0.031 |
| kvp vs h2o | 256 | 4 | 60 | 3 | 9 | 0.15 | 1 |
| kvp vs h2o | 256 | 8 | 60 | 0 | 2 | 0.5 | 1 |
| kvp vs streaming | 256 | 2 | 60 | 7 | 12 | 0.36 | 1 |
| kvp vs streaming | 256 | 4 | 60 | 3 | 1 | 0.62 | 1 |
| kvp vs streaming | 256 | 8 | 60 | 0 | 0 | 1 | 1 |
| streaming_first64 vs kvp | 256 | 2 | 60 | 3 | 10 | 0.092 | 0.74 |
| streaming_first64 vs kvp | 256 | 4 | 60 | 0 | 3 | 0.25 | 1 |
| streaming_first64 vs kvp | 256 | 8 | 60 | 0 | 0 | 1 | 1 |

## Pre-registered pooled test (k in [4, 8], not adjusted)

| A vs B | b | n | only A right | only B right | p |
|---|---|---|---|---|---|
| kvp vs h2o | 256 | 120 | 3 | 11 | 0.057 |
| kvp vs streaming | 256 | 120 | 3 | 1 | 0.62 |
| streaming_first64 vs kvp | 256 | 120 | 0 | 3 | 0.25 |
