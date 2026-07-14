# Top 10 Beta 616 — Experimental Challenger Audit

## Decision

Top 10 Beta 616 is the only tested candidate with enough local upside to make
the current top-10 cutoff plausible. It remains a higher-uncertainty choice
than Challenger 530 and is packaged separately. No live submission was made.

The strategy keeps the validated residual lead-lag network and changes only
two small structural details:

1. The alpha residual uses a 25% adjustment toward expanding heterogeneous
   ALGO betas, retaining 75% of the stable beta-one prior.
2. The 60-day reversal contribution inside the 20% reversal sleeve increases
   from 7.5% to 10%.

The beta adjustment ramps from zero at 200 return observations to its full
value at 300 observations. The live evaluator already supplies at least the
500-day training history, so the mature specification is active immediately.

## Local comparison

| Metric | Challenger 530 | Top 10 Beta 616 |
|---|---:|---:|
| Score | 530.90 | **615.71** |
| Mean P&L | $549.76 | **$631.68** |
| StdDev P&L | $1,638.37 | **$1,608.60** |
| Annualized Sharpe | 5.31 | **6.21** |
| Early score | **381.60** | 366.09 |
| Positive 50-day folds | 8/8 | 8/8 |
| Median 50-day fold | 438.24 | **579.95** |
| Worst 50-day fold | **127.19** | 93.73 |
| Median pseudo-hidden window | 449.49 | **512.76** |
| Worst pseudo-hidden window | 360.53 | **367.41** |
| Last 200 days | 482.05 | **575.72** |
| Last 150 days | 522.20 | **608.41** |
| Last 100 days | 651.74 | **730.32** |
| 10x commission score | 149.73 | **241.67** |
| Strict frozen score | **282.89** | 275.53 |

## Statistical evidence

- Exact position-level causal IC: **0.05267 / 0.05760** for Challenger / Beta
  616.
- Mean paired IC improvement: **0.00493**; ordinary paired p-value **0.0354**.
- 20-day moving-block IC-difference 5th/50th/95th percentiles:
  **0.00213 / 0.00584 / 0.00959**.
- Block-bootstrap probability of positive IC improvement: **99.61%**.
- Realized-P&L moving-block score-difference 5th/50th/95th percentiles:
  **+67.90 / +110.50 / +149.43**; Beta 616 wins **100%**.
- The improvement changes a median of only **2 of 50** synthetic directions
  per day, but 33 of 50 constituent P&L contributions improve.
- Seven nearby beta/long-reversal configurations score above 600 and 22 score
  above 590, so the result is not isolated to one exact pair of parameters.

## Adverse evidence

- A strict graph and beta fit frozen at day 249 is slightly worse than
  Challenger: **275.53 versus 282.89**.
- On block-bootstrapped price paths appended after all 500 observations, the
  median score gain is only **6.88** with a **56%** win rate.
- In no-overlap half-sample path bootstraps, win rates are **50%** and **47%**
  with median differences **-0.9** and **-5.7**. These simulations do not
  reproduce the observed causal IC improvement.
- With observation noise of 1/5/10/20 bps, Beta 616 beats Challenger in
  **92% / 58% / 70% / 56%** of trials. The sharp graph threshold remains
  sensitive to perturbed observations.

The correct interpretation is higher upside with meaningful model risk—not a
guaranteed 615-point hidden score.

## Rejected alternatives in this round

- Volatility-scaled historical standardization: no causal IC improvement.
- Exact 25-long/25-short balancing: weak early and pseudo-hidden results.
- Reintroducing the old two-pair sleeve: local gains but a negative early
  fold.
- Held-out component and target calibration: materially worse than the fixed
  blend.
- Newest-observation overweighting and threshold ensembles: unstable across
  folds.
- Nonlinear nearest-neighbour analogue forecasts: weaker IC and expected edge.
- Lower graph thresholds: strong shuffled-block results but lower chronological
  IC, directional accuracy, and raw expected edge.
- Search of the official public repositories: no disclosed data generator or
  legitimate construction shortcut.

## Contract verification

- ZIP contains exactly root-level `negative alpha.py`.
- Exactly 51 integer positions; one net position per instrument.
- Dynamic $100,000 ALGO and $10,000 constituent caps.
- NaN, infinity, non-positive price, zero-volatility and zero-denominator
  safeguards.
- Same-day idempotency and evaluator reset handling.
- Only NumPy; no `requirements.txt` needed.
- Exact research/submission position identity across all 500 histories: PASS.
- Runtime: approximately **0.15 seconds** for 250 calls through a constructed
  1,750-day history.

## Leaderboard interpretation

At the 14 July 2026 snapshot, Negative Alpha was #19 at 564.14 and the top-10
cutoff was 661.82. If the historical local-to-hidden uplift transferred
additively, Beta 616 would land around 665–672, approximately ranks 8–10 at
that snapshot. That transfer is uncertain and must not be treated as a score
promise.
