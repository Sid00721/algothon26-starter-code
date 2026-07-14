# Lead-Lag Challenger Audit — 14 July 2026

## Decision candidate

Keep the validated one-day diffuse lead-lag network. Add only two restrained
changes:

1. Mix 7.5% of a 60-day residual reversal into the existing reversal sleeve.
2. Calculate the ALGO hedge from causal 120-day asset betas while leaving the
   alpha residualization unchanged.

The proven live submission remains untouched in `negative alpha.zip` and at
git commit `4ff6734`. The challenger is packaged separately.

## Public last-250 comparison

| Model | Score | Mean P&L | StdDev P&L | Sharpe | Volume |
|---|---:|---:|---:|---:|---:|
| Live baseline | 507.42 | $526.38 | $1,608.82 | 5.17 | $108.23m |
| Challenger | **530.90** | **$549.76** | $1,638.37 | **5.31** | $107.59m |

## Temporal robustness

| Test | Live baseline | Challenger |
|---|---:|---:|
| Early-period score | 364.81 | **381.60** |
| Positive 50-day folds | 8/8 | 8/8 |
| Median 50-day fold | 430.23 | **438.24** |
| Worst 50-day fold | 116.20 | **127.19** |
| Median 250-day pseudo-hidden window | 422.45 | **449.49** |
| Worst 250-day pseudo-hidden window | 326.06 | **360.53** |
| Frozen-through-day-249 score | **290.57** | 282.89 |
| Score at 10x commissions | 123.46 | **149.73** |

The challenger improves every listed temporal measure except the strict frozen
network test. Nearby long-reversal weights and beta-shrinkage settings also
improve, so the result is not isolated to one exact parameter combination.

## Paired stress tests

- Moving-block bootstrap score-difference 5th/50th/95th percentiles:
  **-10.39 / 10.94 / 34.49**.
- Bootstrap probability challenger beats baseline: **79.8%**.
- Price-noise trials favor the challenger in **82–90%** of simulations from
  1 to 20 bps observation noise.
- At 20 bps noise, baseline/challenger median scores: **375.05 / 389.13**.
- Daily P&L correlation: **0.9821**.
- Incremental daily P&L mean/std: **$23.37 / $308.38**.
- Median/max synthetic direction changes versus baseline: **0 / 3 of 50 per
  day**. Most of the change is the beta-weighted ALGO hedge.
- Integer, cap, idempotency, ZIP-import and exact research-position identity
  checks on the final artifact: **PASS**.
- Sequential runtime over all 500 histories: **0.078 seconds**.

## What was rejected

- Two-to-five-day lead propagation: higher last-window result in one variant,
  but negative earlier folds.
- Direct ALGO-leading and bidirectional factor timing: weaker than baseline.
- Sparse top-leader networks: the predictive edge is diffuse.
- Directional-asymmetry filters: removed useful structure.
- Low-rank SVD networks: unstable across regimes.
- Recent-window and edge-consensus filters: lower scores and commission
  resilience.
- Dropping historically weak assets: all 50 assets improve aggregate mean in
  forward selection checks.

## Remaining risk

The leaderboard evaluates submissions on a growing reservoir of unseen data.
No public backtest reveals the challenger's hidden score. Its improvement is
credible but modest, and the paired bootstrap still gives roughly a one-in-five
chance of underperforming the live model on a resampled public period.
