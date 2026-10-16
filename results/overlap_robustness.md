## KVP vs baselines without the 27 dialogues whose filler occurs in training

| b | k | baseline | subset | n | KVP acc | baseline acc | only KVP right | only baseline right | p (exact McNemar) |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 2 | h2o | all | 60 | 0.867 | 0.417 | 27 | 0 | 1.5e-08 |
| 256 | 2 | h2o | no overlap | 54 | 0.852 | 0.407 | 24 | 0 | 1.2e-07 |
| 256 | 2 | streaming | all | 60 | 0.867 | 0.400 | 28 | 0 | 7.5e-09 |
| 256 | 2 | streaming | no overlap | 54 | 0.852 | 0.352 | 27 | 0 | 1.5e-08 |
| 256 | 2 | h2o_std | all | 60 | 0.867 | 0.217 | 39 | 0 | 3.6e-12 |
| 256 | 2 | h2o_std | no overlap | 54 | 0.852 | 0.204 | 35 | 0 | 5.8e-11 |
| 256 | 4 | h2o | all | 60 | 0.333 | 0.100 | 15 | 1 | 0.00052 |
| 256 | 4 | h2o | no overlap | 54 | 0.315 | 0.093 | 13 | 1 | 0.0018 |
| 256 | 4 | streaming | all | 60 | 0.333 | 0.033 | 18 | 0 | 7.6e-06 |
| 256 | 4 | streaming | no overlap | 54 | 0.315 | 0.019 | 16 | 0 | 3.1e-05 |
| 256 | 4 | h2o_std | all | 60 | 0.333 | 0.100 | 15 | 1 | 0.00052 |
| 256 | 4 | h2o_std | no overlap | 54 | 0.315 | 0.093 | 13 | 1 | 0.0018 |
| 128 | 2 | h2o | all | 60 | 0.233 | 0.050 | 13 | 2 | 0.0074 |
| 128 | 2 | h2o | no overlap | 54 | 0.259 | 0.019 | 13 | 0 | 0.00024 |
| 128 | 2 | streaming | all | 60 | 0.233 | 0.000 | 14 | 0 | 0.00012 |
| 128 | 2 | streaming | no overlap | 54 | 0.259 | 0.000 | 14 | 0 | 0.00012 |
| 128 | 2 | h2o_std | all | 60 | 0.233 | 0.050 | 13 | 2 | 0.0074 |
| 128 | 2 | h2o_std | no overlap | 54 | 0.259 | 0.019 | 13 | 0 | 0.00024 |
| 128 | 4 | h2o | all | 60 | 0.033 | 0.083 | 1 | 4 | 0.38 |
| 128 | 4 | h2o | no overlap | 54 | 0.037 | 0.074 | 1 | 3 | 0.62 |
| 128 | 4 | streaming | all | 60 | 0.033 | 0.000 | 2 | 0 | 0.5 |
| 128 | 4 | streaming | no overlap | 54 | 0.037 | 0.000 | 2 | 0 | 0.5 |
| 128 | 4 | h2o_std | all | 60 | 0.033 | 0.050 | 1 | 2 | 1 |
| 128 | 4 | h2o_std | no overlap | 54 | 0.037 | 0.037 | 1 | 1 | 1 |
