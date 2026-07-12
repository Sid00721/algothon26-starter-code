# Independent Claude Quant Review

Claude Code 2.1.149 reviewed the evaluator, raw data, generated tables, and representative charts in read-only mode. This is a second opinion, not ground truth; recommendations still need walk-forward testing.

## Main conclusions

- Cross-sectional mean reversion is the strongest observed signal. The 20-day reversal information coefficient is about 0.022 with a t-statistic of 2.74, while longer 80–120 day reversal signals remain positive.
- The first principal component explains about 22.7% of standardized-return variance and has broadly positive loadings, consistent with a shared market factor.
- Individual daily-return autocorrelations are weak and unstable. Claude recommends rejecting single-asset RSI, MACD, and moving-average rules as predictive models unless walk-forward tests prove otherwise.
- Most log-price ADF tests fail to reject a unit root, so strategies should not assume that individual price levels revert to fixed means.
- The proposed data-generating structure is a one-factor process with heterogeneous drift and volatility plus a weak cross-sectional mean-reverting component.

## Recommended candidate design

1. Blend cross-sectional reversal signals over 20 and 80 days.
2. Scale positions using recent volatility.
3. Remove or hedge common-factor exposure.
4. Test slower rebalancing and no-trade bands to reduce turnover.
5. Use ALGO carefully as a low-fee hedge because its position limit is ten times larger.

Claude proposed a starting blend of 70% 20-day reversal and 30% 80-day reversal, with five-day rebalancing. That exact recipe must be challenged against daily and alternative-horizon implementations in walk-forward tests.

## Validation warning

Use non-overlapping forward folds and compare the distribution of fold scores, not only the pooled final-250-day score. Do not select parameters solely because they maximize the visible official backtest. Technical dashboards are diagnostic; visually attractive indicator patterns are not evidence of forward predictability.

## Risks highlighted

- The starter algorithm accumulates positions indefinitely from a one-day signal and has no meaningful risk or turnover control.
- Per-asset drift estimates are noisy with only 500 observations.
- Trying many technical indicators, horizons, and assets creates a severe multiple-testing problem.
- The competition score rewards stable P&L, so uncontrolled common-factor exposure can erase an otherwise useful cross-sectional signal.
