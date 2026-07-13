# Index Statistical-Arbitrage Review — 13 July 2026

## Decision

Submit the aggregated ALGO-neutral residual lead-lag model in
`negative alpha.zip`.  The index/basket OU thesis is rejected, and the
highest visible-score PCA/beta variants are rejected for temporal instability.

The selected strategy is not yesterday's reversal-and-two-pairs algorithm.
It uses all 50 synthetic assets as both predictors and targets, removes the
contemporaneous ALGO return, estimates a sparse cross-asset one-day lead-lag
network, adds a small residual-reversal sleeve, and hedges the realised net
factor exposure with ALGO.

## What the data says about the proposed index trade

- ALGO and the equal-weight basket have **0.9930 contemporaneous daily-return
  correlation**.  This establishes a common factor, not a tradable stationary
  price spread.
- Full-sample log-level cointegration p-value: **0.6162**.
- ALGO/basket basis ADF p-value: **0.9651**.
- Rolling cointegration is significant in only roughly **4–5%** of tested
  windows.
- Index/basket OU forecast correlations range from about **-0.073 to +0.022**
  across 60–250-day specifications.
- An ALGO-to-next-residual predictor reduces the local score from **507.42 to
  483.48**.  Adding a reverse basket-to-ALGO sleeve reduces it further.

The attached Avellaneda–Lee paper is relevant for its factor-residual and OU
framework, but it does not make the observed ALGO/basket basis stationary.
Its useful lesson here is to trade factor-neutral residual structure only after
testing mean reversion speed and stability.

## Causal model comparison

| Family | Local score | Early score | Positive 50-day folds | Minimum fold | Decision |
|---|---:|---:|---:|---:|---|
| Selected residual lead-lag | 507.42 | 364.81 | 8/8 | 116.20 | Submit |
| Lead-lag plus ALGO predictor | 483.48 | 339.68 | 8/8 | 184.45 | Reject: index path dilutes alpha |
| Beta-estimated residual lead-lag | 537.21 | 155.72 | 8/8 | 0.50 | Reject: weak earlier stability |
| PCA residual lead-lag, 2 factors | 520.99 | 0.55 | 6/8 | -164.38 | Reject: regime-dependent |
| Ridge residual VAR | 450.72 | 211.01 | 7/8 | -61.85 | Reject: dense/noisy |
| PCA residual OU | 47–103 | mostly negative | 5–6/8 | negative | Reject |
| ALGO/equal-basket OU | -5.60–2.53 | near zero | 2–5/8 | negative | Reject |

The unhedged lead-lag model and small PCA blends score about **540–542** on the
visible last 250 days.  They were not selected: the extra performance comes
from uncompensated ALGO exposure, the PCA sleeve weakens early/fold behavior,
and a strict fit-through-day-249 test favors the theoretically correct full
factor hedge.  Choosing the visible maximum would repeat the overfitting error
that the hidden leaderboard exposed.

## Evidence for the selected signal

- Expanding causal cross-sectional information coefficient: **0.0484**, with
  standard error **0.0073**.
- Day-permutation one-sided p-value: **0.0005**.
- Chronological 50-day score folds: **8/8 positive**, median **430.23**,
  minimum **116.20**.
- Strictly frozen fit through day 249: pure lead-lag score **344.06**; selected
  20% reversal blend **290.57**.
- At 10x official commission rates: score **123.46**.
- With 20 bps independent observation noise: score 5th/50th/95th percentiles
  **290.04 / 392.00 / 472.59**.

The 20% reversal component helps the expanding causal model but is less robust
under a frozen fit.  It is therefore deliberately a minority sleeve.  The
lead-lag network remains the core.

## Official local evaluation of the exact promoted code

| Score | Mean daily P&L | StdDev daily P&L | Annualized Sharpe | Dollar volume |
|---:|---:|---:|---:|---:|
| 507.42 | $526.38 | $1,608.83 | 5.17 | $108.23m |

These figures are from the supplied `prices.txt`; they are not a promised live
leaderboard result.  The competition evaluates on a different, unseen future
dataset.  That explains why the prior submission could score **7.4647** live
despite replaying much better on the public file: local replay and leaderboard
evaluation are different samples.

## Submission-contract checks

- Root-level file name: `negative alpha.py`.
- Global `getMyPosition(prcSoFar)` function: present.
- Output: exactly 51 integers.
- Dynamic position caps: $100,000 ALGO and $10,000 per synthetic asset.
- Single net position per asset: yes.
- Only NumPy imported; no `requirements.txt` required.
- No network, file, or external-data access.
- ZIP extraction and isolated import: pass.
- Same-day idempotency and cap checks: pass.
- Sequential runtime over all 500 histories: **0.065 seconds**, versus the
  competition's 600-second limit.
- Local activity: **$108.23m**, versus the Testing Round's $25,000 minimum.

## Residual risk

There are only 500 public observations for a 50-asset system, and the hidden
period can change regime.  No backtest eliminates that risk.  The selected
model is the best-supported submission because its signal is causal,
cross-sectionally diversified, statistically detectable, legally packaged,
and less dependent on one visually attractive relationship than the rejected
alternatives.
