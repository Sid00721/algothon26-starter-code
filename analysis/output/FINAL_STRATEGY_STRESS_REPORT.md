# Final Strategy Robustness Report

## Visible official-window result

| Strategy | Score | Mean P&L | StdDev P&L | Annualized Sharpe | Dollar volume |
|---|---:|---:|---:|---:|---:|
| Previous | 212.74 | $292.60 | $2834.69 | 1.63 | $37.16m |
| Final | 355.67 | $413.03 | $2622.61 | 2.49 | $49.70m |

The final candidate improves visible score by 67.2% and lowers daily P&L volatility by 7.5%.

## Robustness summary

- Competition contract, integer dtype, dynamic cap, same-day idempotency, and price-shock checks: **PASS**.
- Sequential calls over all 500 histories: **0.031 seconds** locally.
- Non-overlapping 50-day folds: **8/8 positive**, median score **295.81**, minimum **0.00**.
- At 10x official commissions: score **186.65**.
- 20-day moving-block bootstrap score 5th/50th/95th percentiles: **99.92 / 347.99 / 577.38**; positive probability **99.9%**.
- With 20 bps independent price-observation noise, score 5th/50th/95th percentiles: **308.99 / 341.94 / 366.11**.
- Removing both pairs reduces score to **260.54**. AENO/NWIG alone scores **312.95** and HUXZ/ACAC alone scores **303.98**.
- One-at-a-time sensitivity range across 26 variants: **260.54 to 369.39**.

## Interpretation

The evidence supports a compact ensemble: broad 5/20-day volatility-scaled reversal supplies diversified alpha, while two independently useful spread relationships improve stability. The selected settings sit on a broad performance plateau rather than at an isolated optimum. Exact ALGO beta hedging and the earlier 60/40 residual model were excluded because their ablation results materially reduced score.

This is still a 500-observation research sample, not proof of hidden-period performance. Pair selection and parameter selection use the supplied sample, so the remaining primary risk is regime change and multiple-testing overfit.
