# Batch v5: exploratory analyses (not pre-registered)

Descriptive only; not part of the rule verdicts.

## A. Pooling at b=256, digit sets

Paired against kvp_ft on the same dialogues. needle_at_answer = share of needle tokens kept (mean over heads); needle_heads_full_at_answer = share of heads holding every needle token.

### seed-0 (results/logs_eval_v4)

| config | k | n | acc | only pooled right | only kvp_ft right | needle_at_answer | needle_heads_full_at_answer |
|---|---|---|---|---|---|---|---|
| kvp_ft | 2 | 60 | 0.950 | - | - | 0.812 | 0.621 |
| kvp_ft | 4 | 60 | 0.450 | - | - | 0.360 | 0.142 |
| kvp_ft | 8 | 60 | 0.033 | - | - | 0.093 | 0.017 |
| kvp_ft_pool4 | 2 | 60 | 0.983 | 3 | 1 | 0.819 | 0.611 |
| kvp_ft_pool4 | 4 | 60 | 0.217 | 3 | 17 | 0.398 | 0.127 |
| kvp_ft_pool4 | 8 | 60 | 0.033 | 1 | 1 | 0.118 | 0.013 |
| kvp_ft_pool8 | 2 | 60 | 1.000 | 3 | 0 | 0.909 | 0.834 |
| kvp_ft_pool8 | 4 | 60 | 0.550 | 13 | 7 | 0.491 | 0.343 |
| kvp_ft_pool8 | 8 | 60 | 0.017 | 1 | 2 | 0.215 | 0.119 |

### lead-2 (results/logs_eval_v4_lead2)

| config | k | n | acc | only pooled right | only kvp_ft right | needle_at_answer | needle_heads_full_at_answer |
|---|---|---|---|---|---|---|---|
| kvp_ft | 2 | 60 | 0.367 | - | - | 0.402 | 0.162 |
| kvp_ft | 4 | 60 | 0.133 | - | - | 0.199 | 0.037 |
| kvp_ft | 8 | 60 | 0.000 | - | - | 0.074 | 0.007 |
| kvp_ft_pool4 | 2 | 60 | 0.450 | 13 | 8 | 0.460 | 0.182 |
| kvp_ft_pool4 | 4 | 60 | 0.067 | 3 | 7 | 0.234 | 0.045 |
| kvp_ft_pool4 | 8 | 60 | 0.000 | 0 | 0 | 0.103 | 0.007 |
| kvp_ft_pool8 | 2 | 60 | 0.350 | 11 | 12 | 0.546 | 0.397 |
| kvp_ft_pool8 | 4 | 60 | 0.033 | 1 | 7 | 0.289 | 0.189 |
| kvp_ft_pool8 | 8 | 60 | 0.000 | 0 | 0 | 0.130 | 0.081 |

## B. Words vs digits at matched length

Words: 175 / 180 dialogues the full cache answered correctly. Digits: lead-2 set, all dialogues. Bins of 150 tokens of total_len.

### kvp, b=256

| total_len | words n | words acc | words needle_at_answer | digits n | digits acc | digits needle_at_answer |
|---|---|---|---|---|---|---|
| 150-299 | 0 | - | - | 2 | 1.000 | 0.944 |
| 300-449 | 7 | 0.857 | 0.661 | 10 | 0.700 | 0.539 |
| 450-599 | 28 | 0.286 | 0.467 | 44 | 0.136 | 0.334 |
| 600-749 | 37 | 0.081 | 0.341 | 37 | 0.054 | 0.216 |
| 750-899 | 37 | 0.000 | 0.215 | 20 | 0.000 | 0.139 |
| 900-1049 | 17 | 0.000 | 0.174 | 17 | 0.000 | 0.090 |
| 1050-1199 | 13 | 0.000 | 0.120 | 24 | 0.000 | 0.056 |
| 1200-1349 | 23 | 0.000 | 0.106 | 19 | 0.000 | 0.057 |
| 1350-1499 | 12 | 0.000 | 0.094 | 6 | 0.000 | 0.046 |
| 1500-1649 | 1 | 0.000 | 0.056 | 1 | 0.000 | 0.042 |

### kvp_ft, b=256

| total_len | words n | words acc | words needle_at_answer | digits n | digits acc | digits needle_at_answer |
|---|---|---|---|---|---|---|
| 150-299 | 0 | - | - | 2 | 1.000 | 0.942 |
| 300-449 | 7 | 0.714 | 0.656 | 11 | 0.818 | 0.547 |
| 450-599 | 25 | 0.240 | 0.443 | 42 | 0.262 | 0.357 |
| 600-749 | 40 | 0.125 | 0.352 | 36 | 0.194 | 0.242 |
| 750-899 | 37 | 0.027 | 0.221 | 24 | 0.042 | 0.158 |
| 900-1049 | 17 | 0.000 | 0.159 | 15 | 0.000 | 0.103 |
| 1050-1199 | 17 | 0.000 | 0.112 | 25 | 0.000 | 0.074 |
| 1200-1349 | 21 | 0.000 | 0.113 | 17 | 0.000 | 0.074 |
| 1350-1499 | 8 | 0.000 | 0.086 | 6 | 0.000 | 0.053 |
| 1500-1649 | 3 | 0.000 | 0.085 | 2 | 0.000 | 0.056 |

### Words: name vs city (conditioned on full:0 correct)

| method | b | k | n | name only | city only | both | neither |
|---|---|---|---|---|---|---|---|
| h2o | 256 | 2 | 56 | 0.018 | 0.125 | 0.714 | 0.143 |
| h2o | 256 | 4 | 59 | 0.051 | 0.136 | 0.627 | 0.186 |
| h2o | 256 | 8 | 60 | 0.050 | 0.200 | 0.383 | 0.367 |
| kvp | 256 | 2 | 56 | 0.036 | 0.321 | 0.304 | 0.339 |
| kvp | 256 | 4 | 59 | 0.017 | 0.186 | 0.000 | 0.797 |
| kvp | 256 | 8 | 60 | 0.000 | 0.000 | 0.000 | 1.000 |
| kvp_ft | 128 | 2 | 56 | 0.018 | 0.071 | 0.018 | 0.893 |
| kvp_ft | 128 | 4 | 59 | 0.000 | 0.000 | 0.000 | 1.000 |
| kvp_ft | 128 | 8 | 60 | 0.000 | 0.000 | 0.000 | 1.000 |
| kvp_ft | 256 | 2 | 56 | 0.054 | 0.375 | 0.268 | 0.304 |
| kvp_ft | 256 | 4 | 59 | 0.017 | 0.169 | 0.034 | 0.780 |
| kvp_ft | 256 | 8 | 60 | 0.000 | 0.000 | 0.000 | 1.000 |
| kvp_ft_pool4 | 128 | 2 | 56 | 0.000 | 0.036 | 0.000 | 0.964 |
| kvp_ft_pool4 | 128 | 4 | 59 | 0.000 | 0.000 | 0.000 | 1.000 |
| kvp_ft_pool4 | 128 | 8 | 60 | 0.000 | 0.000 | 0.000 | 1.000 |
| kvp_ft_pool4 | 256 | 2 | 56 | 0.000 | 0.411 | 0.161 | 0.429 |
| kvp_ft_pool4 | 256 | 4 | 59 | 0.000 | 0.085 | 0.000 | 0.915 |
| kvp_ft_pool4 | 256 | 8 | 60 | 0.000 | 0.000 | 0.000 | 1.000 |
| kvp_ft_pool8 | 128 | 2 | 56 | 0.000 | 0.036 | 0.018 | 0.946 |
| kvp_ft_pool8 | 128 | 4 | 59 | 0.000 | 0.000 | 0.000 | 1.000 |
| kvp_ft_pool8 | 128 | 8 | 60 | 0.000 | 0.000 | 0.000 | 1.000 |
| kvp_ft_pool8 | 256 | 2 | 56 | 0.000 | 0.286 | 0.179 | 0.536 |
| kvp_ft_pool8 | 256 | 4 | 59 | 0.000 | 0.102 | 0.000 | 0.898 |
| kvp_ft_pool8 | 256 | 8 | 60 | 0.000 | 0.000 | 0.000 | 1.000 |
| streaming | 256 | 2 | 56 | 0.018 | 0.036 | 0.214 | 0.732 |
| streaming | 256 | 4 | 59 | 0.000 | 0.000 | 0.000 | 1.000 |
| streaming | 256 | 8 | 60 | 0.000 | 0.000 | 0.000 | 1.000 |

## C. Echoes of the needle in the full-cache reply to the needle turn

Each dialogue is grouped by whether the full-cache model repeated the needle when it answered the needle turn (case-insensitive substring of the decoded reply; digits: the 5-digit value). Accuracy of each method at b=256 per group. Words conditioned on full:0 correct; digits = lead-2 (full cache 180/180).

### words (q1, conditioned on full:0 correct)

Full-cache replies to the needle turn: name+city echoed 85, one echoed 48, none echoed 42 (of 175).

| method | b | k | name+city: correct / n | one: correct / n | none: correct / n |
|---|---|---|---|---|---|
| kvp | 256 | 2 | 17 / 33 | 0 / 12 | 0 / 11 |
| kvp | 256 | 4 | 0 / 27 | 0 / 19 | 0 / 13 |
| kvp | 256 | 8 | 0 / 25 | 0 / 17 | 0 / 18 |
| kvp | 256 | all | 17 / 85 | 0 / 48 | 0 / 42 |
| h2o | 256 | 2 | 32 / 33 | 7 / 12 | 1 / 11 |
| h2o | 256 | 4 | 25 / 27 | 12 / 19 | 0 / 13 |
| h2o | 256 | 8 | 14 / 25 | 9 / 17 | 0 / 18 |
| h2o | 256 | all | 71 / 85 | 28 / 48 | 1 / 42 |
| streaming | 256 | 2 | 10 / 33 | 0 / 12 | 2 / 11 |
| streaming | 256 | 4 | 0 / 27 | 0 / 19 | 0 / 13 |
| streaming | 256 | 8 | 0 / 25 | 0 / 17 | 0 / 18 |
| streaming | 256 | all | 10 / 85 | 0 / 48 | 2 / 42 |
| kvp_ft | 256 | 2 | 13 / 33 | 2 / 12 | 0 / 11 |
| kvp_ft | 256 | 4 | 2 / 27 | 0 / 19 | 0 / 13 |
| kvp_ft | 256 | 8 | 0 / 25 | 0 / 17 | 0 / 18 |
| kvp_ft | 256 | all | 15 / 85 | 2 / 48 | 0 / 42 |

### digits lead-2

Full-cache replies to the needle turn: code echoed 15, none echoed 165 (of 180).

| method | b | k | code: correct / n | none: correct / n |
|---|---|---|---|---|
| kvp | 256 | 2 | 2 / 6 | 12 / 54 |
| kvp | 256 | 4 | 0 / 7 | 3 / 53 |
| kvp | 256 | 8 | 0 / 2 | 0 / 58 |
| kvp | 256 | all | 2 / 15 | 15 / 165 |
| h2o | 256 | 2 | 2 / 6 | 1 / 54 |
| h2o | 256 | 4 | 7 / 7 | 2 / 53 |
| h2o | 256 | 8 | 2 / 2 | 0 / 58 |
| h2o | 256 | all | 11 / 15 | 3 / 165 |
| streaming | 256 | 2 | 3 / 6 | 16 / 54 |
| streaming | 256 | 4 | 0 / 7 | 1 / 53 |
| streaming | 256 | 8 | 0 / 2 | 0 / 58 |
| streaming | 256 | all | 3 / 15 | 17 / 165 |
| kvp_ft | 256 | 2 | 4 / 6 | 18 / 54 |
| kvp_ft | 256 | 4 | 2 / 7 | 6 / 53 |
| kvp_ft | 256 | 8 | 0 / 2 | 0 / 58 |
| kvp_ft | 256 | all | 6 / 15 | 24 / 165 |
