#!/usr/bin/env python3
"""Paired robustness audit for the live strategy and its restrained challenger."""

from __future__ import annotations

import time
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.leadlag_improvement_lab import Config, ResearchLeadLag
from analysis.stress_final_strategy import backtest, competition_score


TABLES = ROOT / "analysis" / "output" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

BASELINE = Config()
CHALLENGER = replace(
    BASELINE,
    long_reversal_lookback=60,
    long_reversal_weight=0.075,
    hedge_beta_window=120,
    hedge_beta_shrinkage=1.0,
)


def contract_checks(prices, strategy):
    limits = np.full(51, 10_000.0)
    limits[0] = 100_000.0
    for days in (1, 20, 59, 60, 61, 120, 250, 500):
        strategy.resetState()
        output = strategy.getMyPosition(prices[:, :days])
        repeat = strategy.getMyPosition(prices[:, :days])
        share_limits = (limits / prices[:, days - 1]).astype(int)
        assert output.shape == (51,)
        assert np.issubdtype(output.dtype, np.integer)
        assert np.array_equal(output, repeat)
        assert np.all(np.abs(output) <= share_limits)

    strategy.resetState()
    started = time.perf_counter()
    for days in range(1, prices.shape[1] + 1):
        strategy.getMyPosition(prices[:, :days])
    return time.perf_counter() - started


def position_comparison(prices):
    baseline = ResearchLeadLag(BASELINE)
    challenger = ResearchLeadLag(CHALLENGER)
    direction_differences = []
    algo_dollar_differences = []
    for day in range(249, 499):
        first = baseline.getMyPosition(prices[:, : day + 1])
        second = challenger.getMyPosition(prices[:, : day + 1])
        direction_differences.append(np.sum(np.sign(first[1:]) != np.sign(second[1:])))
        algo_dollar_differences.append(abs(first[0] - second[0]) * prices[0, day])
    return np.asarray(direction_differences), np.asarray(algo_dollar_differences)


def main():
    prices = pd.read_csv(ROOT / "prices.txt", sep=r"\s+").values.T
    baseline = backtest(prices, ResearchLeadLag(BASELINE), 249, 499)
    challenger = backtest(prices, ResearchLeadLag(CHALLENGER), 249, 499)
    runtime = contract_checks(prices, ResearchLeadLag(CHALLENGER))

    rng = np.random.default_rng(20260714)
    noise_rows = []
    for bps in (1, 5, 10, 20):
        baseline_scores = []
        challenger_scores = []
        for _ in range(50):
            observed = prices * np.exp(rng.normal(0.0, bps * 1e-4, prices.shape))
            baseline_scores.append(backtest(prices, ResearchLeadLag(BASELINE), 249, 499, observed_prices=observed)["score"])
            challenger_scores.append(backtest(prices, ResearchLeadLag(CHALLENGER), 249, 499, observed_prices=observed)["score"])
        baseline_scores = np.asarray(baseline_scores)
        challenger_scores = np.asarray(challenger_scores)
        difference = challenger_scores - baseline_scores
        noise_rows.append(
            {
                "noise_bps": bps,
                "baseline_median": float(np.median(baseline_scores)),
                "challenger_median": float(np.median(challenger_scores)),
                "difference_p05": float(np.quantile(difference, 0.05)),
                "difference_median": float(np.median(difference)),
                "difference_p95": float(np.quantile(difference, 0.95)),
                "challenger_win_fraction": float(np.mean(difference > 0.0)),
            }
        )
    noise = pd.DataFrame(noise_rows)
    noise.to_csv(TABLES / "leadlag_challenger_noise.csv", index=False)

    # Paired moving-block bootstrap retains serial dependence and asks the
    # relevant question: how often does the challenger beat the live model?
    rng = np.random.default_rng(20260714)
    score_differences = []
    block = 20
    for _ in range(10_000):
        starts = rng.integers(0, len(baseline["pnl"]) - block + 1, size=13)
        indices = np.concatenate([np.arange(start, start + block) for start in starts])[:250]
        base_sample = baseline["pnl"][indices]
        challenge_sample = challenger["pnl"][indices]
        base_score = competition_score(float(base_sample.mean()), float(base_sample.std()))
        challenge_score = competition_score(float(challenge_sample.mean()), float(challenge_sample.std()))
        score_differences.append(challenge_score - base_score)
    score_differences = np.asarray(score_differences)
    bootstrap = {
        "p05": float(np.quantile(score_differences, 0.05)),
        "median": float(np.median(score_differences)),
        "p95": float(np.quantile(score_differences, 0.95)),
        "win_fraction": float(np.mean(score_differences > 0.0)),
    }

    direction_differences, algo_dollar_differences = position_comparison(prices)
    daily_difference = challenger["pnl"] - baseline["pnl"]
    report = f"""# Lead-Lag Challenger Audit — 14 July 2026

## Local last-250 comparison

| Model | Score | Mean P&L | StdDev P&L | Sharpe | Volume |
|---|---:|---:|---:|---:|---:|
| Live baseline | {baseline['score']:.2f} | ${baseline['mean_pnl']:.2f} | ${baseline['std_pnl']:.2f} | {baseline['sharpe']:.2f} | ${baseline['volume']/1e6:.2f}m |
| Challenger | {challenger['score']:.2f} | ${challenger['mean_pnl']:.2f} | ${challenger['std_pnl']:.2f} | {challenger['sharpe']:.2f} | ${challenger['volume']/1e6:.2f}m |

## Paired stress results

- Moving-block bootstrap score difference 5th/50th/95th percentiles: **{bootstrap['p05']:.2f} / {bootstrap['median']:.2f} / {bootstrap['p95']:.2f}**.
- Bootstrap probability challenger beats baseline: **{bootstrap['win_fraction']:.1%}**.
- Daily P&L correlation: **{np.corrcoef(baseline['pnl'], challenger['pnl'])[0, 1]:.4f}**.
- Incremental daily P&L mean/std: **${daily_difference.mean():.2f} / ${daily_difference.std():.2f}**.
- Synthetic directions changed per day, median/max: **{np.median(direction_differences):.0f} / {direction_differences.max()} of 50**.
- Absolute ALGO hedge change, median/95th percentile: **${np.median(algo_dollar_differences):,.0f} / ${np.quantile(algo_dollar_differences, .95):,.0f}**.
- Contract, integer, cap, idempotency, and price-history checks: **PASS**.
- Sequential runtime over 500 histories: **{runtime:.3f} seconds**.

## Interpretation

The challenger is a controlled modification, not a replacement architecture.
It keeps the validated one-day diffuse lead-lag network, mixes 7.5% of a
60-day residual reversal into the reversal sleeve, and beta-weights only the
ALGO hedge using a 120-day window.  The live ZIP remains recoverable at git
commit `4ff6734` and should not be overwritten unless the total evidence favors
the challenger.
"""
    print(report)
    print("\nNoise stress\n", noise.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    main()
