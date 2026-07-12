# Algothon 2026 Quantitative EDA

Generated from `prices.txt` with 500 daily observations and 51 assets.

## Headline findings

- Average pairwise daily-return correlation: **0.200**.
- First principal component explains **22.7%** of standardized-return variance.
- Median annualized volatility: **33.5%**.
- Median full-sample Sharpe: **-0.41**.
- Strongest full-sample drift: **OTCS, CUBO, RRES, MMBT, ILVX**.
- Weakest full-sample drift: **FARS, NPCK, EAFC, MHRM, SRNA**.
- 20-day reversal mean next-day cross-sectional IC: **0.022** (t-stat **2.74**).

These are exploratory in-sample statistics, not guarantees of hidden-window performance. Strategy selection should use walk-forward tests.

## Core charts

![All prices](figures/01_all_prices.png)
![Normalized prices](figures/02_normalized_prices.png)
![Log returns](figures/03_log_returns.png)
![Rolling volatility](figures/04_rolling_volatility.png)
![Drawdowns](figures/05_drawdowns.png)
![Covariance](figures/06_covariance_matrix.png)
![Correlation](figures/07_correlation_matrix.png)
![Distributions](figures/08_return_distributions.png)
![Risk return](figures/09_risk_return_map.png)
![Autocorrelation](figures/10_autocorrelation_heatmap.png)
![Rolling market correlation](figures/11_rolling_market_correlation.png)
![PCA](figures/12_pca_structure.png)
![Clustering](figures/13_correlation_dendrogram.png)
![Forward IC](figures/14_forward_ic.png)
![Regimes](figures/15_market_regimes.png)

## Per-asset technical dashboards

The `figures/technical/` directory contains a dashboard for every asset with moving averages, Bollinger Bands, RSI(14), MACD, and rolling annualized volatility.

## Current strategy diagnostics

![Strategy diagnostics](figures/16_strategy_diagnostics.png)
![Asset contributions](figures/17_strategy_asset_contributions.png)
![Position heatmap](figures/18_strategy_exposure_heatmap.png)

Walk-forward results are stored in `tables/walk_forward.csv`. The final visible-window score must be treated as an in-sample diagnostic rather than a hidden-window performance estimate.

## Machine-readable results

The `tables/` directory contains the summary statistics, covariance/correlation matrices, rolling volatility, autocorrelations, PCA results, and forward information coefficients.
