# Needle rank screen (full cache up to the question, single-cut approximation)

data: results/mt_niah.jsonl; agents: base=/workspace/kv/ml-learning-to-evict/agents/grouped/oasst2_0_5b, ft=/workspace/kv/ml-learning-to-evict/agents/grouped/oasst2_recall_ft_0_5b

| agents | k | n | mean len | pct | kept@128 | all5@128 | kept@256 | all5@256 |
|---|---|---|---|---|---|---|---|---|
| base | 2 | 60 | 317 | 0.476 | 0.326 | 0.108 | 0.801 | 0.605 |
| base | 4 | 60 | 540 | 0.407 | 0.092 | 0.021 | 0.335 | 0.120 |
| base | 8 | 60 | 959 | 0.341 | 0.021 | 0.001 | 0.080 | 0.019 |
| ft | 2 | 60 | 317 | 0.494 | 0.351 | 0.123 | 0.816 | 0.625 |
| ft | 4 | 60 | 540 | 0.425 | 0.110 | 0.022 | 0.363 | 0.142 |
| ft | 8 | 60 | 959 | 0.358 | 0.024 | 0.001 | 0.096 | 0.020 |
