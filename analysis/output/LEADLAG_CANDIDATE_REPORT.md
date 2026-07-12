# Aggregated Lead-Lag Candidate Audit

The candidate remains separate from `teamName.py`; no submission file was replaced.

## Direct comparison on local days 251-500

| Strategy | Score | Mean P&L | StdDev P&L | Sharpe |
|---|---:|---:|---:|---:|
| Failed submitted strategy | 355.67 | $413.03 | $2622.61 | 2.49 |
| Lead-lag candidate | 507.42 | $526.38 | $1608.83 | 5.17 |

## Evidence

- Expanding causal cross-sectional IC: **0.0484**, standard error **0.0073**.
- Chronological IC chunks: **0.002 / 0.033 / 0.043 / 0.065 / 0.054 / 0.057 / 0.047 / 0.049 / 0.096**; **9/9 positive**.
- Day-permutation null one-sided p-value: **0.0005**.
- Selected blend: official score **507.42**, early score **364.81**, **8/8 positive folds**.
- Pure lead-lag: official score **457.29**, early score **186.20**, **8/8 positive folds**.
- Strict fit-through-day-249 then frozen test score: **344.06** for pure lead-lag and **290.57** with the 20% reversal sleeve.
- 10x commission score: **123.46**.
- 20 bps observation-noise score 5th/50th/95th percentiles: **290.04 / 392.00 / 472.59**.
- Contract and cap checks: **PASS**. Sequential 500-history runtime: **0.064s**.

## Decision

The aggregated lead-lag family deserves promotion to a submission candidate. The lead-lag core has materially stronger and more temporally consistent predictive evidence than the failed own-history reversal model. The 20% reversal blend raises the expanding score, but its weaker frozen-fit result means it should be treated as an optional second-stage sleeve rather than assumed alpha.

This report does not claim a hidden score. All price-based tests still use the supplied 500-day sample.
