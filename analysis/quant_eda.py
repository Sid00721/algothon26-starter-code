#!/usr/bin/env python3
"""Generate a reproducible quantitative EDA report for Algothon prices."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller


TRADING_DAYS = 250


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=155, bbox_inches="tight")
    plt.close()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / window, adjust=False).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + relative_strength)


def technical_frame(price: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"price": price})
    for window in (20, 50, 100):
        frame[f"ma{window}"] = price.rolling(window).mean()
    mean20 = frame["ma20"]
    std20 = price.rolling(20).std()
    frame["bb_upper"] = mean20 + 2 * std20
    frame["bb_lower"] = mean20 - 2 * std20
    frame["rsi14"] = rsi(price)
    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    frame["macd"] = ema12 - ema26
    frame["macd_signal"] = frame["macd"].ewm(span=9, adjust=False).mean()
    frame["vol20"] = np.log(price).diff().rolling(20).std() * np.sqrt(TRADING_DAYS)
    return frame


def plot_asset_grid(data: pd.DataFrame, title: str, ylabel: str, path: Path) -> None:
    ncols = 3
    nrows = int(np.ceil(data.shape[1] / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 2.25 * nrows), sharex=True)
    axes = np.asarray(axes).ravel()
    for ax, column in zip(axes, data.columns):
        ax.plot(data.index, data[column], linewidth=0.75)
        ax.set_title(column, fontsize=9)
        ax.grid(alpha=0.2)
    for ax in axes[data.shape[1] :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15, y=1.002)
    fig.supylabel(ylabel)
    fig.tight_layout()
    savefig(path)


def plot_heatmap(matrix: pd.DataFrame, title: str, path: Path, cmap: str, symmetric: bool) -> None:
    fig, ax = plt.subplots(figsize=(14, 12))
    limit = float(np.nanmax(np.abs(matrix.values))) if symmetric else None
    image = ax.imshow(
        matrix.values,
        cmap=cmap,
        aspect="auto",
        vmin=-limit if symmetric else None,
        vmax=limit if symmetric else None,
    )
    ticks = np.arange(len(matrix.columns))
    ax.set_xticks(ticks, matrix.columns, rotation=90, fontsize=6)
    ax.set_yticks(ticks, matrix.index, fontsize=6)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=0.75)
    savefig(path)


def forward_ic(prices: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    next_returns = np.log(prices.shift(-1) / prices)
    records: list[dict[str, float | int | str]] = []
    for horizon in horizons:
        trailing = np.log(prices / prices.shift(horizon))
        for direction, signal in (("momentum", trailing), ("reversal", -trailing)):
            daily_ic = trailing.copy() * np.nan
            values = []
            for day in range(horizon, len(prices) - 1):
                value = np.corrcoef(signal.iloc[day], next_returns.iloc[day])[0, 1]
                values.append(value)
            array = np.asarray(values)
            records.append(
                {
                    "signal": direction,
                    "horizon": horizon,
                    "mean_ic": float(np.nanmean(array)),
                    "ic_std": float(np.nanstd(array)),
                    "positive_fraction": float(np.nanmean(array > 0)),
                    "t_stat": float(np.nanmean(array) / (np.nanstd(array) / np.sqrt(len(array)))),
                }
            )
    return pd.DataFrame(records)


def create_report(prices_path: Path, output: Path) -> None:
    figures = output / "figures"
    tables = output / "tables"
    technical = figures / "technical"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    technical.mkdir(parents=True, exist_ok=True)

    prices = pd.read_csv(prices_path, sep=r"\s+")
    prices.index.name = "day"
    returns = np.log(prices).diff()
    valid_returns = returns.dropna()
    normalized = prices / prices.iloc[0]
    rolling_vol = returns.rolling(20).std() * np.sqrt(TRADING_DAYS)
    drawdown = prices / prices.cummax() - 1
    covariance = valid_returns.cov() * TRADING_DAYS
    correlation = valid_returns.corr()

    summary = pd.DataFrame(
        {
            "start_price": prices.iloc[0],
            "end_price": prices.iloc[-1],
            "total_log_return": np.log(prices.iloc[-1] / prices.iloc[0]),
            "annual_return": valid_returns.mean() * TRADING_DAYS,
            "annual_volatility": valid_returns.std() * np.sqrt(TRADING_DAYS),
            "full_sample_sharpe": np.sqrt(TRADING_DAYS) * valid_returns.mean() / valid_returns.std(),
            "skew": valid_returns.skew(),
            "excess_kurtosis": valid_returns.kurt(),
            "worst_drawdown": drawdown.min(),
            "lag1_autocorrelation": valid_returns.apply(lambda x: x.autocorr(1)),
        }
    )
    summary["adf_pvalue_log_price"] = [adfuller(np.log(prices[c]), autolag="AIC")[1] for c in prices]
    summary.to_csv(tables / "asset_summary.csv")
    covariance.to_csv(tables / "annualized_covariance.csv")
    correlation.to_csv(tables / "correlation.csv")
    rolling_vol.to_csv(tables / "rolling_20d_annualized_volatility.csv")

    plot_asset_grid(prices, "All asset price series", "Price", figures / "01_all_prices.png")
    plot_asset_grid(normalized, "All prices normalized to day 0", "Growth of 1", figures / "02_normalized_prices.png")
    plot_asset_grid(returns, "Daily log returns", "Log return", figures / "03_log_returns.png")
    plot_asset_grid(rolling_vol, "20-day rolling annualized volatility", "Annualized volatility", figures / "04_rolling_volatility.png")
    plot_asset_grid(drawdown, "Drawdown from running peak", "Drawdown", figures / "05_drawdowns.png")
    plot_heatmap(covariance, "Annualized covariance matrix", figures / "06_covariance_matrix.png", "coolwarm", True)
    plot_heatmap(correlation, "Return correlation matrix", figures / "07_correlation_matrix.png", "coolwarm", True)

    fig, ax = plt.subplots(figsize=(13, 7))
    standardized = (valid_returns - valid_returns.mean()) / valid_returns.std()
    parts = ax.violinplot([standardized[c].dropna() for c in prices], showmeans=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_alpha(0.55)
    ax.set_xticks(np.arange(1, len(prices.columns) + 1), prices.columns, rotation=90, fontsize=6)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("Standardized daily log return")
    ax.set_title("Return distributions and tail behaviour")
    savefig(figures / "08_return_distributions.png")

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(summary["annual_volatility"], summary["annual_return"], c=summary["full_sample_sharpe"], cmap="coolwarm")
    for ticker, row in summary.iterrows():
        ax.annotate(ticker, (row["annual_volatility"], row["annual_return"]), fontsize=6, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized mean log return")
    ax.set_title("Risk-return map")
    fig.colorbar(scatter, ax=ax, label="Full-sample Sharpe")
    savefig(figures / "09_risk_return_map.png")

    lags = np.arange(1, 41)
    autocorrelation = pd.DataFrame(
        {ticker: [valid_returns[ticker].autocorr(int(lag)) for lag in lags] for ticker in prices},
        index=lags,
    )
    autocorrelation.to_csv(tables / "return_autocorrelation.csv", index_label="lag")
    fig, ax = plt.subplots(figsize=(14, 8))
    image = ax.imshow(autocorrelation.T, cmap="coolwarm", aspect="auto", vmin=-0.2, vmax=0.2)
    ax.set_yticks(np.arange(len(prices.columns)), prices.columns, fontsize=6)
    ax.set_xticks(np.arange(0, len(lags), 2), lags[::2])
    ax.set_xlabel("Lag in days")
    ax.set_title("Daily return autocorrelation by asset and lag")
    fig.colorbar(image, ax=ax, label="Autocorrelation")
    savefig(figures / "10_autocorrelation_heatmap.png")

    rolling_market_corr = valid_returns.rolling(60).corr(valid_returns.mean(axis=1))
    plot_asset_grid(rolling_market_corr, "60-day correlation with equal-weight market return", "Correlation", figures / "11_rolling_market_correlation.png")

    pca = PCA().fit(standardized)
    pca_loadings = pd.DataFrame(
        pca.components_.T,
        index=prices.columns,
        columns=[f"PC{i + 1}" for i in range(len(prices.columns))],
    )
    pca_loadings.to_csv(tables / "pca_loadings.csv")
    pd.Series(pca.explained_variance_ratio_, index=pca_loadings.columns, name="explained_variance_ratio").to_csv(tables / "pca_explained_variance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(np.arange(1, 16), pca.explained_variance_ratio_[:15])
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Explained variance ratio")
    axes[0].set_title("PCA eigenvalue spectrum")
    pca_loadings.iloc[:, :3].plot.bar(ax=axes[1], width=0.85)
    axes[1].set_title("First three PCA loadings")
    axes[1].tick_params(axis="x", labelsize=6)
    fig.tight_layout()
    savefig(figures / "12_pca_structure.png")

    distance = np.sqrt(np.maximum(0, (1 - correlation.values) / 2))
    hierarchy = linkage(squareform(distance, checks=False), method="average")
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(hierarchy, labels=prices.columns.tolist(), leaf_rotation=90, leaf_font_size=7, ax=ax)
    ax.set_title("Hierarchical clustering from return correlations")
    ax.set_ylabel("Correlation distance")
    savefig(figures / "13_correlation_dendrogram.png")

    ic = forward_ic(prices, (1, 2, 3, 5, 10, 20, 40, 80, 120))
    ic.to_csv(tables / "forward_information_coefficients.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    for signal, frame in ic.groupby("signal"):
        ax.plot(frame["horizon"], frame["mean_ic"], marker="o", label=signal)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Signal lookback (days)")
    ax.set_ylabel("Mean cross-sectional next-day IC")
    ax.set_title("Momentum versus reversal predictive relationship")
    ax.legend()
    savefig(figures / "14_forward_ic.png")

    fig, ax = plt.subplots(figsize=(12, 5))
    dispersion = valid_returns.std(axis=1)
    market_return = valid_returns.mean(axis=1)
    ax.plot(market_return.index, market_return.rolling(20).std() * np.sqrt(TRADING_DAYS), label="Market-factor volatility")
    ax.plot(dispersion.index, dispersion.rolling(20).mean() * np.sqrt(TRADING_DAYS), label="Cross-sectional dispersion")
    ax.set_title("Market volatility and cross-sectional opportunity")
    ax.set_ylabel("Annualized scale")
    ax.legend()
    savefig(figures / "15_market_regimes.png")

    for ticker in prices:
        frame = technical_frame(prices[ticker])
        fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1, 1]})
        axes[0].plot(frame.index, frame["price"], label="Price", linewidth=1.1)
        axes[0].plot(frame.index, frame["ma20"], label="MA20", linewidth=0.8)
        axes[0].plot(frame.index, frame["ma50"], label="MA50", linewidth=0.8)
        axes[0].plot(frame.index, frame["ma100"], label="MA100", linewidth=0.8)
        axes[0].fill_between(frame.index, frame["bb_lower"], frame["bb_upper"], alpha=0.13, label="Bollinger ±2σ")
        axes[0].set_title(f"{ticker}: price, trend and technical indicators")
        axes[0].legend(ncol=5, fontsize=8)
        axes[1].plot(frame.index, frame["rsi14"], linewidth=0.9)
        axes[1].axhline(70, color="red", linewidth=0.7, linestyle="--")
        axes[1].axhline(30, color="green", linewidth=0.7, linestyle="--")
        axes[1].set_ylabel("RSI(14)")
        axes[2].plot(frame.index, frame["macd"], label="MACD", linewidth=0.9)
        axes[2].plot(frame.index, frame["macd_signal"], label="Signal", linewidth=0.8)
        axes[2].axhline(0, color="black", linewidth=0.6)
        axes[2].legend(fontsize=8)
        axes[2].set_ylabel("MACD")
        axes[3].plot(frame.index, frame["vol20"], linewidth=0.9)
        axes[3].set_ylabel("20d ann. vol")
        axes[3].set_xlabel("Day")
        for axis in axes:
            axis.grid(alpha=0.2)
        fig.tight_layout()
        savefig(technical / f"{ticker}.png")

    top_positive = summary.nlargest(5, "full_sample_sharpe").index.tolist()
    top_negative = summary.nsmallest(5, "full_sample_sharpe").index.tolist()
    mean_pair_corr = (correlation.values.sum() - len(correlation)) / (len(correlation) * (len(correlation) - 1))
    reversal20 = ic[(ic.signal == "reversal") & (ic.horizon == 20)].iloc[0]
    report = f"""# Algothon 2026 Quantitative EDA

Generated from `{prices_path.name}` with {len(prices)} daily observations and {prices.shape[1]} assets.

## Headline findings

- Average pairwise daily-return correlation: **{mean_pair_corr:.3f}**.
- First principal component explains **{pca.explained_variance_ratio_[0]:.1%}** of standardized-return variance.
- Median annualized volatility: **{summary.annual_volatility.median():.1%}**.
- Median full-sample Sharpe: **{summary.full_sample_sharpe.median():.2f}**.
- Strongest full-sample drift: **{', '.join(top_positive)}**.
- Weakest full-sample drift: **{', '.join(top_negative)}**.
- 20-day reversal mean next-day cross-sectional IC: **{reversal20.mean_ic:.3f}** (t-stat **{reversal20.t_stat:.2f}**).

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
"""
    (output / "REPORT.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", type=Path, default=Path("prices.txt"))
    parser.add_argument("--output", type=Path, default=Path("analysis/output"))
    args = parser.parse_args()
    create_report(args.prices, args.output)


if __name__ == "__main__":
    main()
