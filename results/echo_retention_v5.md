# Echo retention and head coverage (v5, exploratory, not pre-registered)

Words: conditioned on the full cache answering correctly (175 / 180). All original-copy numbers were recomputed from the kept indices and match the logged fields.

### Echoes, words

Copy held whole = all its tokens in that (layer, head) at answer time. Echo = the fact repeated in an assistant reply of the SAME run before the question.

| config | n | echoed (all parts) | acc, echoed | acc, not echoed | name orig whole | name echo whole | name orig tok | name echo tok | city orig whole | city echo whole | city orig tok | city echo tok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kvp_b256 | 175 | 87 | 14 / 87 | 3 / 88 | 0.106 | 0.091 | 0.517 | 0.379 | 0.135 | 0.093 | 0.342 | 0.230 |
| h2o_b256 | 175 | 89 | 74 / 89 | 26 / 86 | 0.147 | 0.042 | 0.874 | 0.447 | 0.200 | 0.051 | 0.485 | 0.191 |
| streaming_b256 | 175 | 90 | 8 / 90 | 4 / 85 | 0.063 | 0.091 | 0.063 | 0.091 | 0.063 | 0.063 | 0.069 | 0.063 |
| kvp_ft_b256 | 175 | 86 | 15 / 86 | 2 / 89 | 0.101 | 0.093 | 0.520 | 0.382 | 0.127 | 0.087 | 0.334 | 0.225 |
| kvp_ft_pool4_b256 | 175 | 88 | 8 / 88 | 1 / 87 | 0.138 | 0.198 | 0.187 | 0.267 | 0.299 | 0.191 | 0.373 | 0.232 |
| kvp_ft_pool8_b256 | 175 | 89 | 9 / 89 | 1 / 86 | 0.142 | 0.213 | 0.251 | 0.264 | 0.322 | 0.167 | 0.342 | 0.181 |

name recalled, by which copy of the name is held whole in at least one head:

| config | original + echo | original only | echo only | neither |
|---|---|---|---|---|
| kvp_b256 | 17 / 83 | 3 / 64 | 0 / 9 | 0 / 19 |
| h2o_b256 | 69 / 72 | 38 / 85 | 0 / 0 | 0 / 18 |
| streaming_b256 | 7 / 7 | 4 / 4 | 2 / 9 | 0 / 155 |
| kvp_ft_b256 | 18 / 84 | 3 / 62 | 0 / 9 | 0 / 20 |
| kvp_ft_pool4_b256 | 9 / 116 | 0 / 51 | 0 / 4 | 0 / 4 |
| kvp_ft_pool8_b256 | 9 / 112 | 1 / 52 | 0 / 4 | 0 / 7 |

### Echoes, digits

Copy held whole = all its tokens in that (layer, head) at answer time. Echo = the fact repeated in an assistant reply of the SAME run before the question.

| config | n | echoed (all parts) | acc, echoed | acc, not echoed | code orig whole | code echo whole | code orig tok | code echo tok |
|---|---|---|---|---|---|---|---|---|
| kvp_ft_b256 | 180 | 21 | 7 / 21 | 23 / 159 | 0.069 | 0.007 | 0.431 | 0.055 |
| kvp_ft_pool4_b256 | 180 | 25 | 6 / 25 | 25 / 155 | 0.078 | 0.018 | 0.382 | 0.057 |
| kvp_ft_pool8_b256 | 180 | 18 | 5 / 18 | 18 / 162 | 0.222 | 0.022 | 0.378 | 0.028 |

code recalled, by which copy of the code is held whole in at least one head:

| config | original + echo | original only | echo only | neither |
|---|---|---|---|---|
| kvp_ft_b256 | 7 / 14 | 23 / 122 | 0 / 3 | 0 / 41 |
| kvp_ft_pool4_b256 | 6 / 15 | 25 / 101 | 0 / 8 | 0 / 56 |
| kvp_ft_pool8_b256 | 5 / 18 | 18 / 161 | 0 / 0 | 0 / 1 |

### Head coverage, words

| config | n | mean survival | heads whole | heads any | any / mean |
|---|---|---|---|---|---|
| kvp_b256 | 175 | 0.265 | 0.054 | 0.579 | 2.18 |
| h2o_b256 | 175 | 0.406 | 0.077 | 0.887 | 2.18 |
| streaming_b256 | 175 | 0.064 | 0.063 | 0.069 | 1.07 |
| kvp_ft_b256 | 175 | 0.262 | 0.049 | 0.577 | 2.20 |
| kvp_ft_pool4_b256 | 175 | 0.228 | 0.085 | 0.419 | 1.84 |
| kvp_ft_pool8_b256 | 175 | 0.251 | 0.122 | 0.379 | 1.51 |

Per mean-survival bin:

| config | survival bin | n | correct | heads whole | heads any |
|---|---|---|---|---|---|
| kvp_b256 | [0, 0.2) | 79 | 0 | 0.009 | 0.377 |
| kvp_b256 | [0.2, 0.4) | 63 | 4 | 0.047 | 0.677 |
| kvp_b256 | [0.4, 0.6) | 26 | 8 | 0.136 | 0.856 |
| kvp_b256 | [0.6, 1.0) | 7 | 5 | 0.310 | 0.952 |
| h2o_b256 | [0, 0.2) | 20 | 0 | 0.003 | 0.645 |
| h2o_b256 | [0.2, 0.4) | 69 | 25 | 0.024 | 0.874 |
| h2o_b256 | [0.4, 0.6) | 67 | 57 | 0.095 | 0.949 |
| h2o_b256 | [0.6, 1.0) | 19 | 18 | 0.281 | 0.973 |
| streaming_b256 | [0, 0.2) | 164 | 1 | 0.000 | 0.006 |
| streaming_b256 | [0.6, 1.0) | 11 | 11 | 1.000 | 1.000 |
| kvp_ft_b256 | [0, 0.2) | 74 | 0 | 0.007 | 0.370 |
| kvp_ft_b256 | [0.2, 0.4) | 69 | 6 | 0.046 | 0.666 |
| kvp_ft_b256 | [0.4, 0.6) | 26 | 8 | 0.123 | 0.848 |
| kvp_ft_b256 | [0.6, 1.0) | 6 | 3 | 0.281 | 0.951 |
| kvp_ft_pool4_b256 | [0, 0.2) | 94 | 0 | 0.031 | 0.246 |
| kvp_ft_pool4_b256 | [0.2, 0.4) | 57 | 3 | 0.111 | 0.554 |
| kvp_ft_pool4_b256 | [0.4, 0.6) | 21 | 5 | 0.208 | 0.752 |
| kvp_ft_pool4_b256 | [0.6, 1.0) | 3 | 1 | 0.403 | 0.903 |
| kvp_ft_pool8_b256 | [0, 0.2) | 87 | 0 | 0.041 | 0.191 |
| kvp_ft_pool8_b256 | [0.2, 0.4) | 49 | 2 | 0.138 | 0.457 |
| kvp_ft_pool8_b256 | [0.4, 0.6) | 34 | 5 | 0.248 | 0.681 |
| kvp_ft_pool8_b256 | [0.6, 1.0) | 5 | 3 | 0.500 | 0.829 |

### Head coverage, digits

| config | n | mean survival | heads whole | heads any | any / mean |
|---|---|---|---|---|---|
| kvp_ft_b256 | 180 | 0.225 | 0.069 | 0.431 | 1.92 |
| kvp_ft_pool4_b256 | 180 | 0.266 | 0.078 | 0.382 | 1.44 |
| kvp_ft_pool8_b256 | 180 | 0.322 | 0.222 | 0.378 | 1.17 |

Per mean-survival bin:

| config | survival bin | n | correct | heads whole | heads any |
|---|---|---|---|---|---|
| kvp_ft_b256 | [0, 0.2) | 94 | 2 | 0.013 | 0.254 |
| kvp_ft_b256 | [0.2, 0.4) | 63 | 13 | 0.074 | 0.558 |
| kvp_ft_b256 | [0.4, 0.6) | 17 | 9 | 0.205 | 0.765 |
| kvp_ft_b256 | [0.6, 1.0) | 6 | 6 | 0.503 | 0.917 |
| kvp_ft_pool4_b256 | [0, 0.2) | 87 | 0 | 0.009 | 0.207 |
| kvp_ft_pool4_b256 | [0.2, 0.4) | 53 | 6 | 0.075 | 0.443 |
| kvp_ft_pool4_b256 | [0.4, 0.6) | 29 | 16 | 0.172 | 0.624 |
| kvp_ft_pool4_b256 | [0.6, 1.0) | 11 | 9 | 0.386 | 0.845 |
| kvp_ft_pool8_b256 | [0, 0.2) | 65 | 0 | 0.077 | 0.153 |
| kvp_ft_pool8_b256 | [0.2, 0.4) | 51 | 2 | 0.182 | 0.335 |
| kvp_ft_pool8_b256 | [0.4, 0.6) | 46 | 9 | 0.331 | 0.572 |
| kvp_ft_pool8_b256 | [0.6, 1.0) | 18 | 12 | 0.584 | 0.812 |

