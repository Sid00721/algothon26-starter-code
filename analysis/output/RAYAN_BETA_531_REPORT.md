# Rayan Beta 531 — Structural Refinement Audit

## Verdict

`Rayan Beta 531` is the separately labelled comparison strategy requested by
Rayan's prompt. It retains Challenger 530's validated alpha model but uses a
60-day beta hedge instead of 120 days. It is legal, fast and marginally higher
on the clean public window. It is **not** sufficiently superior to replace
Challenger 530 on robustness evidence.

## Side-by-side result

| Metric | Challenger 530 | Rayan Beta 531 |
|---|---:|---:|
| Score | 530.90 | **531.02** |
| Mean P&L | $549.76 | **$549.83** |
| StdDev P&L | $1,638.37 | **$1,635.95** |
| Sharpe | 5.31 | **5.31** |
| Early score | 381.60 | **386.65** |
| Median fold | **438.24** | 437.72 |
| Worst fold | 127.19 | **127.46** |
| Positive folds | 8/8 | 8/8 |
| Median 250-day window | **449.49** | 446.02 |
| Worst 250-day window | 360.53 | **361.86** |
| 10x commission score | **149.73** | 149.65 |

## Paired uncertainty

- Moving-block bootstrap score difference, Rayan minus Challenger,
  5th/50th/95th percentiles: **-14.71 / -1.75 / 10.82**.
- Bootstrap probability Rayan Beta 531 beats Challenger 530: **41.1%**.
- Across 1/5/10/20 bps price-noise tests, median score differences are
  **+0.64 / +2.58 / +1.36 / +0.18**.
- At 20 bps observation noise, the win rate is exactly **50%**.

The point estimate is effectively a tie. Challenger 530 has the better median
bootstrap and median 250-day-window evidence.

## Prompt assumptions tested

### L1/PCA replacement

- Multi-task L1 alpha 0.03: score **420.35**, mean **$444.69**, StdDev
  **$1,691.95**, runtime **3.06 seconds** for the public 250-day evaluation.
- Multi-task L1 alpha 0.10: score **479.64**, mean **$501.14**, StdDev
  **$1,677.62**, runtime **1.18 seconds**.
- Weak L1 penalties emitted convergence warnings and made the full causal grid
  take minutes.
- Proximal/soft-threshold L1 approximation: score **327.37**.
- Best two-factor PCA lead-lag: score **520.99**, but early score **0.55**,
  only **6/8** positive folds and worst fold **-164.38**.

L1 and PCA are rejected. Exact sparsity and collinearity handling do not
compensate for their loss of stable diffuse signal.

### Volatility-scaled hysteresis

The best tested version scores **506.37**, below Challenger 530's **530.90**.
Most versions score 462–500. Residual standardization already accounts for
per-asset scale, so applying volatility again to the deadband double-counts
volatility and suppresses profitable flips.

### Expanding versus rolling estimation

- Second-half/first-half residual variance ratio: **0.9873**.
- Levene and Fligner variance-stability p-values: **0.2585 / 0.2453**.
- Rolling 60-day residual variance remains between **0.918x and 1.078x** its
  median.
- No 25%, 50% or 100% recent/prior variance-break trigger fires.

There is no evidence supporting a forced 250-day rolling window. Expanding
estimation is retained to reduce coefficient variance.

## Engineering checks

- NumPy-vectorized alpha and hedge calculations.
- NaN/non-positive price forward-fill and first-valid backfill included.
- Zero-variance and zero-beta-denominator safeguards included.
- No matrix inversion is used, so singular-matrix Ridge fallback is unnecessary.
- Dynamic 51-instrument integer caps retained.
- Deterministic 1,750-day runtime test: **0.130 seconds** for the 250 evaluation
  calls using extended histories.
- Final ZIP isolated import and clean-data reference identity: **PASS**.
- Injected NaN, Inf, all-invalid-row, zero-volatility and idempotency checks:
  **PASS**.
- Exact final-artifact 1,750-day runtime: **0.133 seconds** for 250 calls.

## Recommendation

Use this artifact for comparison and explanation. If choosing one submission,
prefer **Challenger 530** because its improvement over the live model is
material, whereas Rayan Beta 531's improvement over Challenger 530 is within
sampling noise.
