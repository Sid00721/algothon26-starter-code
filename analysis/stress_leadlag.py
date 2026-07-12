#!/usr/bin/env python3
"""Reproducible audit of the clean-room lead-lag candidate."""

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

from analysis.stress_final_strategy import backtest


TABLES = ROOT / "analysis" / "output" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def contract_checks(prices, strategy):
    limits = np.full(51, 10_000.0)
    limits[0] = 100_000.0
    for days in (1, 20, 59, 60, 61, 120, 250, 500):
        strategy.resetState()
        first = strategy.getMyPosition(prices[:, :days])
        repeat = strategy.getMyPosition(prices[:, :days])
        share_limits = (limits / prices[:, days - 1]).astype(int)
        assert first.shape == (51,)
        assert np.issubdtype(first.dtype, np.integer)
        assert np.array_equal(first, repeat)
        assert np.all(np.abs(first) <= share_limits)

    strategy.resetState()
    started = time.perf_counter()
    for days in range(1, prices.shape[1] + 1):
        strategy.getMyPosition(prices[:, :days])
    return time.perf_counter() - started


def causal_ic(prices, significance_se=1.0, minimum=60):
    returns = np.diff(np.log(prices), axis=1)
    residuals = returns[1:] - returns[0]
    signals = []
    outcomes = []
    daily_ic = []
    for current in range(minimum, residuals.shape[1] - 1):
        sample = residuals[:, : current + 1]
        mean = sample.mean(axis=1, keepdims=True)
        scale = np.maximum(sample.std(axis=1, keepdims=True), 1e-12)
        z = (sample - mean) / scale
        predictors = z[:, :-1]
        targets = z[:, 1:]
        n_pairs = predictors.shape[1]
        matrix = predictors @ targets.T / n_pairs
        np.fill_diagonal(matrix, 0.0)
        matrix[np.abs(matrix) < significance_se / np.sqrt(n_pairs)] = 0.0
        signal = matrix.T @ z[:, -1]
        outcome = (residuals[:, current + 1] - mean[:, 0]) / scale[:, 0]
        signals.append(signal)
        outcomes.append(outcome)
        daily_ic.append(np.corrcoef(signal, outcome)[0, 1])
    return np.asarray(signals), np.asarray(outcomes), np.asarray(daily_ic)


def set_config(strategy, original, changes):
    for key, value in original.items():
        setattr(strategy, key, value)
    for key, value in changes.items():
        setattr(strategy, key, value)
    strategy.resetState()


def main():
    frame = pd.read_csv(ROOT / "prices.txt", sep=r"\s+")
    prices = frame.values.T
    candidate = importlib.import_module("analysis.candidate_leadlag")
    failed = importlib.import_module("teamName")

    configurable = (
        "SIGNIFICANCE_SE",
        "REVERSAL_WEIGHT",
        "HYSTERESIS_FRACTION",
        "SIZING_MODE",
        "ESTIMATION_WINDOW",
        "FREEZE_ESTIMATION_AT",
        "HEDGE_FRACTION",
    )
    original = {key: getattr(candidate, key) for key in configurable}
    runtime = contract_checks(prices, candidate)

    comparisons = []
    for label, strategy in (("failed_submission", failed), ("leadlag_candidate", candidate)):
        result = backtest(prices, strategy, 249, 499)
        comparisons.append({"strategy": label, **{key: result[key] for key in ("score", "mean_pnl", "std_pnl", "sharpe", "volume")}})
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(TABLES / "leadlag_comparison.csv", index=False)

    variants = [
        ("selected", {}),
        ("pure_leadlag", {"REVERSAL_WEIGHT": 0.0}),
        ("reversal_0.10", {"REVERSAL_WEIGHT": 0.10}),
        ("reversal_0.15", {"REVERSAL_WEIGHT": 0.15}),
        ("reversal_0.25", {"REVERSAL_WEIGHT": 0.25}),
        ("threshold_0", {"SIGNIFICANCE_SE": 0.0}),
        ("threshold_0.5", {"SIGNIFICANCE_SE": 0.5}),
        ("threshold_1.5", {"SIGNIFICANCE_SE": 1.5}),
        ("threshold_2", {"SIGNIFICANCE_SE": 2.0}),
        ("no_hysteresis", {"HYSTERESIS_FRACTION": 0.0}),
        ("hysteresis_0.15", {"HYSTERESIS_FRACTION": 0.15}),
        ("hysteresis_0.45", {"HYSTERESIS_FRACTION": 0.45}),
        ("tanh_sizing", {"SIZING_MODE": "tanh"}),
        ("rank_sizing", {"SIZING_MODE": "rank"}),
        ("rolling_120", {"ESTIMATION_WINDOW": 120}),
        ("rolling_200", {"ESTIMATION_WINDOW": 200}),
        ("rolling_250", {"ESTIMATION_WINDOW": 250}),
        ("rolling_350", {"ESTIMATION_WINDOW": 350}),
        ("no_ALGO_hedge", {"HEDGE_FRACTION": 0.0}),
        ("half_ALGO_hedge", {"HEDGE_FRACTION": 0.5}),
    ]
    sensitivity_rows = []
    for label, changes in variants:
        set_config(candidate, original, changes)
        early = backtest(prices, candidate, 100, 249)
        official = backtest(prices, candidate, 249, 499)
        fold_scores = [backtest(prices, candidate, start, min(start + 50, 499))["score"] for start in range(100, 499, 50)]
        sensitivity_rows.append(
            {
                "variant": label,
                "official_score": official["score"],
                "official_mean": official["mean_pnl"],
                "official_std": official["std_pnl"],
                "early_score": early["score"],
                "fold_median": float(np.median(fold_scores)),
                "fold_min": float(np.min(fold_scores)),
                "positive_folds": int(np.sum(np.asarray(fold_scores) > 0)),
                "volume": official["volume"],
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(TABLES / "leadlag_sensitivity.csv", index=False)

    frozen_rows = []
    for reversal_weight in (0.0, 0.15, 0.20, 0.25):
        for freeze_at in (120, 180, 249, 300, 350):
            changes = {"REVERSAL_WEIGHT": reversal_weight, "FREEZE_ESTIMATION_AT": freeze_at}
            set_config(candidate, original, changes)
            result = backtest(prices, candidate, 249, 499)
            frozen_rows.append(
                {
                    "reversal_weight": reversal_weight,
                    "freeze_at": freeze_at,
                    **{key: result[key] for key in ("score", "mean_pnl", "std_pnl", "sharpe")},
                }
            )
    frozen = pd.DataFrame(frozen_rows)
    frozen.to_csv(TABLES / "leadlag_frozen_fit.csv", index=False)

    cost_rows = []
    set_config(candidate, original, {})
    for multiplier in (0, 1, 2, 5, 10, 20):
        result = backtest(prices, candidate, 249, 499, multiplier)
        cost_rows.append({"commission_multiplier": multiplier, **{key: result[key] for key in ("score", "mean_pnl", "std_pnl")}})
    costs = pd.DataFrame(cost_rows)
    costs.to_csv(TABLES / "leadlag_cost_stress.csv", index=False)

    rng = np.random.default_rng(20260712)
    noise_rows = []
    for bps in (1, 5, 10, 20):
        scores = []
        for _ in range(50):
            observed = prices * np.exp(rng.normal(0.0, bps * 1e-4, prices.shape))
            scores.append(backtest(prices, candidate, 249, 499, observed_prices=observed)["score"])
        noise_rows.append(
            {
                "noise_bps": bps,
                "score_p05": float(np.quantile(scores, 0.05)),
                "score_median": float(np.median(scores)),
                "score_p95": float(np.quantile(scores, 0.95)),
            }
        )
    noise = pd.DataFrame(noise_rows)
    noise.to_csv(TABLES / "leadlag_price_noise.csv", index=False)

    signals, outcomes, daily_ic = causal_ic(prices)
    chunks = [float(daily_ic[start : start + 50].mean()) for start in range(0, len(daily_ic), 50)]
    rng = np.random.default_rng(2026)
    null_ic = []
    for _ in range(2_000):
        permuted_days = rng.permutation(len(outcomes))
        correlations = [np.corrcoef(signals[day], outcomes[permuted_days[day]])[0, 1] for day in range(len(outcomes))]
        null_ic.append(float(np.mean(correlations)))
    p_value = (1 + np.sum(np.asarray(null_ic) >= daily_ic.mean())) / (1 + len(null_ic))

    selected = sensitivity.loc[sensitivity.variant == "selected"].iloc[0]
    pure = sensitivity.loc[sensitivity.variant == "pure_leadlag"].iloc[0]
    report = f"""# Aggregated Lead-Lag Candidate Audit

The candidate remains separate from `teamName.py`; no submission file was replaced.

## Direct comparison on local days 251-500

| Strategy | Score | Mean P&L | StdDev P&L | Sharpe |
|---|---:|---:|---:|---:|
| Failed submitted strategy | {comparison.iloc[0].score:.2f} | ${comparison.iloc[0].mean_pnl:.2f} | ${comparison.iloc[0].std_pnl:.2f} | {comparison.iloc[0].sharpe:.2f} |
| Lead-lag candidate | {comparison.iloc[1].score:.2f} | ${comparison.iloc[1].mean_pnl:.2f} | ${comparison.iloc[1].std_pnl:.2f} | {comparison.iloc[1].sharpe:.2f} |

## Evidence

- Expanding causal cross-sectional IC: **{daily_ic.mean():.4f}**, standard error **{daily_ic.std(ddof=1)/np.sqrt(len(daily_ic)):.4f}**.
- Chronological IC chunks: **{' / '.join(f'{value:.3f}' for value in chunks)}**; **{sum(value > 0 for value in chunks)}/{len(chunks)} positive**.
- Day-permutation null one-sided p-value: **{p_value:.4f}**.
- Selected blend: official score **{selected.official_score:.2f}**, early score **{selected.early_score:.2f}**, **{int(selected.positive_folds)}/8 positive folds**.
- Pure lead-lag: official score **{pure.official_score:.2f}**, early score **{pure.early_score:.2f}**, **{int(pure.positive_folds)}/8 positive folds**.
- Strict fit-through-day-249 then frozen test score: **{frozen.loc[(frozen.freeze_at == 249) & (frozen.reversal_weight == 0), 'score'].iloc[0]:.2f}** for pure lead-lag and **{frozen.loc[(frozen.freeze_at == 249) & (frozen.reversal_weight == 0.20), 'score'].iloc[0]:.2f}** with the 20% reversal sleeve.
- 10x commission score: **{costs.loc[costs.commission_multiplier == 10, 'score'].iloc[0]:.2f}**.
- 20 bps observation-noise score 5th/50th/95th percentiles: **{noise.iloc[-1].score_p05:.2f} / {noise.iloc[-1].score_median:.2f} / {noise.iloc[-1].score_p95:.2f}**.
- Contract and cap checks: **PASS**. Sequential 500-history runtime: **{runtime:.3f}s**.

## Decision

The aggregated lead-lag family deserves promotion to a submission candidate. The lead-lag core has materially stronger and more temporally consistent predictive evidence than the failed own-history reversal model. The 20% reversal blend raises the expanding score, but its weaker frozen-fit result means it should be treated as an optional second-stage sleeve rather than assumed alpha.

This report does not claim a hidden score. All price-based tests still use the supplied 500-day sample.
"""
    (ROOT / "analysis" / "output" / "LEADLAG_CANDIDATE_REPORT.md").write_text(report)

    set_config(candidate, original, {})
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print("\nSensitivity\n", sensitivity.sort_values("official_score", ascending=False).to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print("\nFrozen fits\n", frozen.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print("\nCosts\n", costs.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print("\nNoise\n", noise.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print("\nIC", f"{daily_ic.mean():.4f}", "p", f"{p_value:.4f}", "chunks", chunks)


if __name__ == "__main__":
    main()
