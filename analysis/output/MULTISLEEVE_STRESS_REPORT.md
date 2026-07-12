# Multi-Sleeve Candidate Stress Test

The proposed 60% residual-reversion / 40% stable-pairs candidate was implemented in `analysis/candidate_multisleeve.py` and tested without replacing `teamName.py`.

## Competition compliance

- Returns exactly 51 integer positions.
- Nets sleeve dollar exposures before conversion to shares.
- Enforces daily price-dependent share caps: $100,000 for ALGO and $10,000 for every other asset.
- Re-clips held positions before applying the deadband, so a price move cannot leave a retained position above its new limit.
- Repeated calls for the same day are idempotent.
- A shorter history resets state cleanly.
- Fewer than 120 days uses a conservative 5-day reversal fallback.
- Uses only NumPy and the `prcSoFar` argument.
- Correct pair indices are AENO/NWIG `(1, 20)` and HUXZ/ACAC `(8, 27)`. Index 30 is RCRI.
- After integer rounding, the median absolute residual factor exposure was about $230 and the maximum about $1,066.

## Official visible-window comparison

| Strategy | Score | Mean P&L | StdDev P&L | Sharpe | Dollar volume |
|---|---:|---:|---:|---:|---:|
| Current `teamName.py` | 212.74 | $292.60 | $2,834.69 | 1.63 | $37.16m |
| Proposed multi-sleeve | 35.21 | $55.00 | $651.95 | 1.33 | $9.83m |

The multi-sleeve candidate substantially reduces volatility, but it sacrifices too much mean P&L for the competition's scoring function.

## Ablations and sensitivity

| Variant | Score | Mean P&L | StdDev P&L |
|---|---:|---:|---:|
| 20/80-day residual reversal | 42.87 | $63.46 | $695.37 |
| Residual sleeve only | 40.19 | $72.66 | $1,032.59 |
| Default 60/40 | 35.21 | $55.00 | $651.95 |
| 40/60 residual/pairs | 34.45 | $47.10 | $451.45 |
| Pairs only | 28.26 | $33.81 | $237.03 |
| Incorrect HUXZ/RCRI index | 25.76 | $46.75 | $667.01 |
| No ALGO hedge | 20.28 | $43.35 | $731.10 |

The ALGO hedge improves the candidate, and the ACAC index correction matters. The best tested version still scores less than one quarter of the current strategy.

## Commission stress

| Commission multiplier | Score | Mean P&L |
|---:|---:|---:|
| 1× | 35.21 | $55.00 |
| 2× | 31.32 | $51.45 |
| 5× | 20.17 | $40.80 |
| 10× | 5.48 | $23.06 |

The deadband keeps the strategy profitable under severe commission increases, but commission is not the main limitation; insufficient gross alpha is.

## Stability

Across seven non-overlapping 50-day walk-forward folds, the candidate had five positive and two negative scores. Its 20-day moving-block bootstrap score distribution was:

- 5th percentile: **-53.73**
- Median: **12.07**
- 95th percentile: **93.06**
- Probability of a positive resampled score: **73.8%**

## Decision

Do not promote this exact candidate to `teamName.py`. Retain the current strategy while researching a higher-capacity residual/pairs design. The useful components to salvage are:

1. Correct ALGO beta hedging.
2. The AENO/NWIG and HUXZ/ACAC pair hypotheses.
3. Net dollar aggregation before share conversion.
4. Daily integer cap enforcement and idempotent state handling.
5. Deadband implementation.

Full machine-readable results are in:

- `multisleeve_sensitivity.csv`
- `multisleeve_walk_forward.csv`
- `multisleeve_cost_stress.csv`
