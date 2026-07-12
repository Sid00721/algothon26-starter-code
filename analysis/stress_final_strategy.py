#!/usr/bin/env python3
"""Reproducible robustness audit for the final Algothon strategy."""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TABLES = ROOT / "analysis" / "output" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def competition_score(mean_pnl, std_pnl):
    if mean_pnl <= 0 or std_pnl < 1e-10:
        return float(mean_pnl)
    sharpe = np.sqrt(250) * mean_pnl / std_pnl
    return float(mean_pnl * sharpe**2 / (sharpe**2 + 1.0))


def backtest(prices, strategy, start, end, commission_multiplier=1.0, observed_prices=None):
    if hasattr(strategy, "resetState"):
        strategy.resetState()
    if observed_prices is None:
        observed_prices = prices

    n_instruments = prices.shape[0]
    dollar_limits = np.full(n_instruments, 10_000.0)
    dollar_limits[0] = 100_000.0
    fee_rates = np.full(n_instruments, 0.0001) * commission_multiplier
    fee_rates[0] = 0.00002 * commission_multiplier
    position = np.zeros(n_instruments, dtype=int)
    pnl = []
    asset_pnl = np.zeros(n_instruments)
    volume = 0.0

    started = time.perf_counter()
    for day in range(start, end):
        current = prices[:, day]
        requested = np.asarray(strategy.getMyPosition(observed_prices[:, : day + 1]))
        share_limits = (dollar_limits / current).astype(int)
        new_position = np.clip(requested, -share_limits, share_limits).astype(int)
        trade = new_position - position
        traded_dollars = current * np.abs(trade)
        components = new_position * (prices[:, day + 1] - current) - traded_dollars * fee_rates
        asset_pnl += components
        pnl.append(float(components.sum()))
        volume += float(traded_dollars.sum())
        position = new_position

    pnl = np.asarray(pnl)
    mean_pnl = float(pnl.mean())
    std_pnl = float(pnl.std())
    sharpe = float(np.sqrt(250) * mean_pnl / std_pnl) if std_pnl else 0.0
    return {
        "score": competition_score(mean_pnl, std_pnl),
        "mean_pnl": mean_pnl,
        "std_pnl": std_pnl,
        "sharpe": sharpe,
        "volume": volume,
        "elapsed": time.perf_counter() - started,
        "pnl": pnl,
        "asset_pnl": asset_pnl,
    }


class BaselineStrategy:
    """The previously submitted 20/80-weighted 5/20-day reversal model."""

    @staticmethod
    def getMyPosition(prcSoFar):
        prices = np.asarray(prcSoFar, dtype=float)
        n_instruments, n_days = prices.shape
        if n_days <= 20:
            return np.zeros(n_instruments, dtype=int)
        start = max(0, n_days - 61)
        volatility = np.maximum(np.std(np.diff(np.log(prices[:, start:])), axis=1, ddof=1), 1e-6)
        five = -np.log(prices[:, -1] / prices[:, -6]) / (volatility * np.sqrt(5))
        twenty = -np.log(prices[:, -1] / prices[:, -21]) / (volatility * np.sqrt(20))
        signal = 0.2 * five + 0.8 * twenty
        limits = np.full(n_instruments, 10_000.0)
        limits[0] = 100_000.0
        share_limits = (limits / prices[:, -1]).astype(int)
        target = np.rint(limits * np.tanh(3.0 * signal) / prices[:, -1]).astype(int)
        return np.clip(target, -share_limits, share_limits).astype(int)


def contract_checks(prices, strategy):
    limits = np.full(prices.shape[0], 10_000.0)
    limits[0] = 100_000.0
    for n_days in (1, 6, 20, 21, 119, 120, 121, 250, 500):
        strategy.resetState()
        output = strategy.getMyPosition(prices[:, :n_days])
        repeat = strategy.getMyPosition(prices[:, :n_days])
        share_limits = (limits / prices[:, n_days - 1]).astype(int)
        assert output.shape == (51,)
        assert np.issubdtype(output.dtype, np.integer)
        assert np.array_equal(output, repeat)
        assert np.all(np.abs(output) <= share_limits)

    for multiplier in (0.1, 10.0):
        shocked = prices[:, :250].copy()
        shocked[:, -1] *= multiplier
        strategy.resetState()
        output = strategy.getMyPosition(shocked)
        share_limits = (limits / shocked[:, -1]).astype(int)
        assert np.all(np.abs(output) <= share_limits)

    strategy.resetState()
    started = time.perf_counter()
    for n_days in range(1, prices.shape[1] + 1):
        strategy.getMyPosition(prices[:, :n_days])
    return time.perf_counter() - started


def metrics_row(label, result):
    return {
        "test": label,
        **{key: result[key] for key in ("score", "mean_pnl", "std_pnl", "sharpe", "volume", "elapsed")},
    }


def main():
    frame = pd.read_csv(ROOT / "prices.txt", sep=r"\s+")
    prices = frame.values.T
    strategy = importlib.import_module("teamName")
    runtime = contract_checks(prices, strategy)

    baseline = backtest(prices, BaselineStrategy, 249, 499)
    final = backtest(prices, strategy, 249, 499)
    comparison = pd.DataFrame([metrics_row("previous", baseline), metrics_row("final", final)])
    comparison.to_csv(TABLES / "final_comparison.csv", index=False)

    folds = []
    for start in range(100, 499, 50):
        end = min(start + 50, 499)
        result = backtest(prices, strategy, start, end)
        folds.append({"start": start + 1, "end": end + 1, **metrics_row("fold", result)})
    fold_frame = pd.DataFrame(folds).drop(columns="test")
    fold_frame.to_csv(TABLES / "final_walk_forward.csv", index=False)

    costs = []
    for multiplier in (0, 1, 2, 5, 10, 20):
        result = backtest(prices, strategy, 249, 499, multiplier)
        costs.append({"commission_multiplier": multiplier, **metrics_row("cost", result)})
    cost_frame = pd.DataFrame(costs).drop(columns="test")
    cost_frame.to_csv(TABLES / "final_cost_stress.csv", index=False)

    configurable = {
        "VOLATILITY_LOOKBACK": strategy.VOLATILITY_LOOKBACK,
        "SHORT_WEIGHT": strategy.SHORT_WEIGHT,
        "MEDIUM_WEIGHT": strategy.MEDIUM_WEIGHT,
        "SIGNAL_STRENGTH": strategy.SIGNAL_STRENGTH,
        "SIGNAL_THRESHOLD": strategy.SIGNAL_THRESHOLD,
        "PAIR_LOOKBACK": strategy.PAIR_LOOKBACK,
        "PAIR_Z_LOOKBACK": strategy.PAIR_Z_LOOKBACK,
        "PAIR_WEIGHT": strategy.PAIR_WEIGHT,
        "PAIRS": strategy.PAIRS,
        "DEADBAND_FRACTION": strategy.DEADBAND_FRACTION,
    }

    variants = [("selected", {})]
    variants += [(f"vol_{value}", {"VOLATILITY_LOOKBACK": value}) for value in (40, 80, 120)]
    variants += [
        (f"short_weight_{value}", {"SHORT_WEIGHT": value, "MEDIUM_WEIGHT": 1 - value})
        for value in (0.4, 0.6)
    ]
    variants += [(f"strength_{value}", {"SIGNAL_STRENGTH": value}) for value in (3.0, 5.0)]
    variants += [(f"threshold_{value}", {"SIGNAL_THRESHOLD": value}) for value in (0.15, 0.35)]
    variants += [(f"pair_weight_{value}", {"PAIR_WEIGHT": value}) for value in (1.0, 2.0, 4.0, 5.0)]
    variants += [(f"pair_lookback_{value}", {"PAIR_LOOKBACK": value}) for value in (80, 100, 160, 200)]
    variants += [(f"pair_z_{value}", {"PAIR_Z_LOOKBACK": value}) for value in (40, 80, 100)]
    variants += [
        ("no_pairs", {"PAIRS": ()}),
        ("AENO_NWIG_only", {"PAIRS": ((1, 20),)}),
        ("HUXZ_ACAC_only", {"PAIRS": ((8, 27),)}),
        ("no_deadband", {"DEADBAND_FRACTION": 0.0}),
        ("wide_deadband", {"DEADBAND_FRACTION": 0.03}),
    ]
    sensitivity_rows = []
    for label, changes in variants:
        for key, value in configurable.items():
            setattr(strategy, key, value)
        for key, value in changes.items():
            setattr(strategy, key, value)
        result = backtest(prices, strategy, 249, 499)
        fold_scores = [backtest(prices, strategy, start, min(start + 50, 499))["score"] for start in range(100, 499, 50)]
        sensitivity_rows.append(
            {
                **metrics_row(label, result),
                "fold_median": float(np.median(fold_scores)),
                "fold_min": float(np.min(fold_scores)),
                "positive_folds": int(np.sum(np.asarray(fold_scores) > 0)),
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(TABLES / "final_sensitivity.csv", index=False)
    for key, value in configurable.items():
        setattr(strategy, key, value)

    rng = np.random.default_rng(20260712)
    noise_rows = []
    for bps in (1, 5, 10, 20):
        scores = []
        for _ in range(100):
            observed = prices * np.exp(rng.normal(0.0, bps * 1e-4, prices.shape))
            scores.append(backtest(prices, strategy, 249, 499, observed_prices=observed)["score"])
        noise_rows.append(
            {
                "noise_bps": bps,
                "score_p05": float(np.quantile(scores, 0.05)),
                "score_median": float(np.median(scores)),
                "score_p95": float(np.quantile(scores, 0.95)),
                "positive_fraction": float(np.mean(np.asarray(scores) > 0)),
            }
        )
    noise_frame = pd.DataFrame(noise_rows)
    noise_frame.to_csv(TABLES / "final_price_noise.csv", index=False)

    # Moving-block bootstrap preserves much more serial dependence than IID sampling.
    rng = np.random.default_rng(2026)
    bootstrap_scores = []
    block_length = 20
    for _ in range(10_000):
        starts = rng.integers(0, len(final["pnl"]) - block_length + 1, size=13)
        sample = np.concatenate([final["pnl"][start : start + block_length] for start in starts])[:250]
        bootstrap_scores.append(competition_score(float(sample.mean()), float(sample.std())))
    bootstrap_quantiles = np.quantile(bootstrap_scores, (0.05, 0.50, 0.95))

    contributions = pd.DataFrame({"instrument": frame.columns, "net_pnl": final["asset_pnl"]})
    contributions["absolute_share"] = contributions.net_pnl.abs() / contributions.net_pnl.abs().sum()
    contributions.sort_values("net_pnl", ascending=False).to_csv(TABLES / "final_asset_contributions.csv", index=False)

    report = f"""# Final Strategy Robustness Report

## Visible official-window result

| Strategy | Score | Mean P&L | StdDev P&L | Annualized Sharpe | Dollar volume |
|---|---:|---:|---:|---:|---:|
| Previous | {baseline['score']:.2f} | ${baseline['mean_pnl']:.2f} | ${baseline['std_pnl']:.2f} | {baseline['sharpe']:.2f} | ${baseline['volume']/1e6:.2f}m |
| Final | {final['score']:.2f} | ${final['mean_pnl']:.2f} | ${final['std_pnl']:.2f} | {final['sharpe']:.2f} | ${final['volume']/1e6:.2f}m |

The final candidate improves visible score by {(final['score']/baseline['score']-1)*100:.1f}% and lowers daily P&L volatility by {(1-final['std_pnl']/baseline['std_pnl'])*100:.1f}%.

## Robustness summary

- Competition contract, integer dtype, dynamic cap, same-day idempotency, and price-shock checks: **PASS**.
- Sequential calls over all 500 histories: **{runtime:.3f} seconds** locally.
- Non-overlapping 50-day folds: **{(fold_frame.score > 0).sum()}/{len(fold_frame)} positive**, median score **{fold_frame.score.median():.2f}**, minimum **{fold_frame.score.min():.2f}**.
- At 10x official commissions: score **{cost_frame.loc[cost_frame.commission_multiplier == 10, 'score'].iloc[0]:.2f}**.
- 20-day moving-block bootstrap score 5th/50th/95th percentiles: **{bootstrap_quantiles[0]:.2f} / {bootstrap_quantiles[1]:.2f} / {bootstrap_quantiles[2]:.2f}**; positive probability **{np.mean(np.asarray(bootstrap_scores) > 0):.1%}**.
- With 20 bps independent price-observation noise, score 5th/50th/95th percentiles: **{noise_frame.iloc[-1].score_p05:.2f} / {noise_frame.iloc[-1].score_median:.2f} / {noise_frame.iloc[-1].score_p95:.2f}**.
- Removing both pairs reduces score to **{sensitivity.loc[sensitivity.test == 'no_pairs', 'score'].iloc[0]:.2f}**. AENO/NWIG alone scores **{sensitivity.loc[sensitivity.test == 'AENO_NWIG_only', 'score'].iloc[0]:.2f}** and HUXZ/ACAC alone scores **{sensitivity.loc[sensitivity.test == 'HUXZ_ACAC_only', 'score'].iloc[0]:.2f}**.
- One-at-a-time sensitivity range across {len(sensitivity)} variants: **{sensitivity.score.min():.2f} to {sensitivity.score.max():.2f}**.

## Interpretation

The evidence supports a compact ensemble: broad 5/20-day volatility-scaled reversal supplies diversified alpha, while two independently useful spread relationships improve stability. The selected settings sit on a broad performance plateau rather than at an isolated optimum. Exact ALGO beta hedging and the earlier 60/40 residual model were excluded because their ablation results materially reduced score.

This is still a 500-observation research sample, not proof of hidden-period performance. Pair selection and parameter selection use the supplied sample, so the remaining primary risk is regime change and multiple-testing overfit.
"""
    (ROOT / "analysis" / "output" / "FINAL_STRATEGY_STRESS_REPORT.md").write_text(report)

    print(comparison.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print("\nWalk-forward\n", fold_frame.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print("\nCommission stress\n", cost_frame.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print("\nNoise stress\n", noise_frame.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print("\nContract checks: PASS; sequential runtime:", f"{runtime:.3f}s")
    print("Bootstrap 5/50/95:", *(f"{value:.2f}" for value in bootstrap_quantiles))


if __name__ == "__main__":
    main()
