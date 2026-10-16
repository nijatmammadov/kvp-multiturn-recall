## Accuracy (95% Wilson CI)

| method | b | k | correct / n | acc | CI |
|---|---|---|---|---|---|
| full | 0 | 2 | 60 / 60 | 1.000 | [0.94, 1.00] |
| full | 0 | 4 | 60 / 60 | 1.000 | [0.94, 1.00] |
| full | 0 | 8 | 60 / 60 | 1.000 | [0.94, 1.00] |
| h2o | 128 | 2 | 3 / 60 | 0.050 | [0.02, 0.14] |
| h2o | 128 | 4 | 5 / 60 | 0.083 | [0.04, 0.18] |
| h2o | 128 | 8 | 1 / 60 | 0.017 | [0.00, 0.09] |
| h2o | 256 | 2 | 25 / 60 | 0.417 | [0.30, 0.54] |
| h2o | 256 | 4 | 6 / 60 | 0.100 | [0.05, 0.20] |
| h2o | 256 | 8 | 5 / 60 | 0.083 | [0.04, 0.18] |
| kvp | 128 | 2 | 14 / 60 | 0.233 | [0.14, 0.35] |
| kvp | 128 | 4 | 2 / 60 | 0.033 | [0.01, 0.11] |
| kvp | 128 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| kvp | 256 | 2 | 52 / 60 | 0.867 | [0.76, 0.93] |
| kvp | 256 | 4 | 20 / 60 | 0.333 | [0.23, 0.46] |
| kvp | 256 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| kvp_ft | 128 | 2 | 22 / 60 | 0.367 | [0.26, 0.49] |
| kvp_ft | 128 | 4 | 2 / 60 | 0.033 | [0.01, 0.11] |
| kvp_ft | 128 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| kvp_ft | 256 | 2 | 57 / 60 | 0.950 | [0.86, 0.98] |
| kvp_ft | 256 | 4 | 27 / 60 | 0.450 | [0.33, 0.58] |
| kvp_ft | 256 | 8 | 2 / 60 | 0.033 | [0.01, 0.11] |
| kvp_rerun | 256 | 2 | 52 / 60 | 0.867 | [0.76, 0.93] |
| kvp_rerun | 256 | 4 | 20 / 60 | 0.333 | [0.23, 0.46] |
| kvp_rerun | 256 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| streaming | 128 | 2 | 0 / 60 | 0.000 | [0.00, 0.06] |
| streaming | 128 | 4 | 0 / 60 | 0.000 | [0.00, 0.06] |
| streaming | 128 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| streaming | 256 | 2 | 24 / 60 | 0.400 | [0.29, 0.53] |
| streaming | 256 | 4 | 2 / 60 | 0.033 | [0.01, 0.11] |
| streaming | 256 | 8 | 0 / 60 | 0.000 | [0.00, 0.06] |
| streaming_first64 | 128 | 2 | 60 / 60 | 1.000 | [0.94, 1.00] |
| streaming_first64 | 128 | 4 | 60 / 60 | 1.000 | [0.94, 1.00] |
| streaming_first64 | 128 | 8 | 60 / 60 | 1.000 | [0.94, 1.00] |
| streaming_first64 | 256 | 2 | 60 / 60 | 1.000 | [0.94, 1.00] |
| streaming_first64 | 256 | 4 | 60 / 60 | 1.000 | [0.94, 1.00] |
| streaming_first64 | 256 | 8 | 60 / 60 | 1.000 | [0.94, 1.00] |

## Paired exact McNemar (30 tests, Holm-adjusted)

| A vs B | b | k | n | only A right | only B right | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| kvp_ft vs kvp | 128 | 2 | 60 | 10 | 2 | 0.039 | 0.5 |
| kvp_ft vs kvp | 128 | 4 | 60 | 0 | 0 | 1 | 1 |
| kvp_ft vs kvp | 128 | 8 | 60 | 0 | 0 | 1 | 1 |
| kvp_ft vs kvp | 256 | 2 | 60 | 5 | 0 | 0.062 | 0.69 |
| kvp_ft vs kvp | 256 | 4 | 60 | 8 | 1 | 0.039 | 0.5 |
| kvp_ft vs kvp | 256 | 8 | 60 | 2 | 0 | 0.5 | 1 |
| kvp_ft vs h2o | 128 | 2 | 60 | 21 | 2 | 6.6e-05 | 0.00099 |
| kvp_ft vs h2o | 128 | 4 | 60 | 1 | 4 | 0.38 | 1 |
| kvp_ft vs h2o | 128 | 8 | 60 | 0 | 1 | 1 | 1 |
| kvp_ft vs h2o | 256 | 2 | 60 | 32 | 0 | 4.7e-10 | 8.8e-09 |
| kvp_ft vs h2o | 256 | 4 | 60 | 22 | 1 | 5.7e-06 | 9.2e-05 |
| kvp_ft vs h2o | 256 | 8 | 60 | 1 | 4 | 0.38 | 1 |
| kvp_ft vs streaming | 128 | 2 | 60 | 22 | 0 | 4.8e-07 | 8.1e-06 |
| kvp_ft vs streaming | 128 | 4 | 60 | 2 | 0 | 0.5 | 1 |
| kvp_ft vs streaming | 128 | 8 | 60 | 0 | 0 | 1 | 1 |
| kvp_ft vs streaming | 256 | 2 | 60 | 33 | 0 | 2.3e-10 | 4.9e-09 |
| kvp_ft vs streaming | 256 | 4 | 60 | 25 | 0 | 6e-08 | 1.1e-06 |
| kvp_ft vs streaming | 256 | 8 | 60 | 2 | 0 | 0.5 | 1 |
| streaming_first64 vs kvp_ft | 128 | 2 | 60 | 38 | 0 | 7.3e-12 | 1.6e-10 |
| streaming_first64 vs kvp_ft | 128 | 4 | 60 | 58 | 0 | 6.9e-18 | 1.9e-16 |
| streaming_first64 vs kvp_ft | 128 | 8 | 60 | 60 | 0 | 1.7e-18 | 5.2e-17 |
| streaming_first64 vs kvp_ft | 256 | 2 | 60 | 3 | 0 | 0.25 | 1 |
| streaming_first64 vs kvp_ft | 256 | 4 | 60 | 33 | 0 | 2.3e-10 | 4.9e-09 |
| streaming_first64 vs kvp_ft | 256 | 8 | 60 | 58 | 0 | 6.9e-18 | 1.9e-16 |
| streaming_first64 vs kvp | 128 | 2 | 60 | 46 | 0 | 2.8e-14 | 6.8e-13 |
| streaming_first64 vs kvp | 128 | 4 | 60 | 58 | 0 | 6.9e-18 | 1.9e-16 |
| streaming_first64 vs kvp | 128 | 8 | 60 | 60 | 0 | 1.7e-18 | 5.2e-17 |
| streaming_first64 vs kvp | 256 | 2 | 60 | 8 | 0 | 0.0078 | 0.11 |
| streaming_first64 vs kvp | 256 | 4 | 60 | 40 | 0 | 1.8e-12 | 4.2e-11 |
| streaming_first64 vs kvp | 256 | 8 | 60 | 60 | 0 | 1.7e-18 | 5.2e-17 |

## Pre-registered pooled test (k in [4, 8], not adjusted)

| A vs B | b | n | only A right | only B right | p |
|---|---|---|---|---|---|
| kvp_ft vs kvp | 128 | 120 | 0 | 0 | 1 |
| kvp_ft vs kvp | 256 | 120 | 10 | 1 | 0.012 |
| kvp_ft vs h2o | 128 | 120 | 1 | 5 | 0.22 |
| kvp_ft vs h2o | 256 | 120 | 23 | 5 | 0.00091 |
| kvp_ft vs streaming | 128 | 120 | 2 | 0 | 0.5 |
| kvp_ft vs streaming | 256 | 120 | 27 | 0 | 1.5e-08 |
| streaming_first64 vs kvp_ft | 128 | 120 | 118 | 0 | 6e-36 |
| streaming_first64 vs kvp_ft | 256 | 120 | 91 | 0 | 8.1e-28 |
| streaming_first64 vs kvp | 128 | 120 | 118 | 0 | 6e-36 |
| streaming_first64 vs kvp | 256 | 120 | 100 | 0 | 1.6e-30 |
