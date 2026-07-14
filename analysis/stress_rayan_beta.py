#!/usr/bin/env python3
"""Paired stress test of Rayan Beta 531 against Challenger 530."""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.leadlag_improvement_lab import ResearchLeadLag
from analysis.rayan_structural_lab import CHALLENGER_530
from analysis.stress_final_strategy import backtest, competition_score


TABLES = ROOT / "analysis" / "output" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
RAYAN_BETA_531 = replace(CHALLENGER_530, hedge_beta_window=60)


def extended_prices(prices, target_days=1_750):
    """Deterministically extend positive prices for runtime testing only."""
    returns = np.diff(np.log(prices), axis=1)
    needed = target_days - 1
    tiled = np.tile(returns, (1, int(np.ceil(needed / returns.shape[1]))))[:, :needed]
    return prices[:, :1] * np.exp(np.cumsum(np.column_stack([np.zeros(prices.shape[0]), tiled]), axis=1))


def main():
    prices = pd.read_csv(ROOT / "prices.txt", sep=r"\s+").values.T
    reference = backtest(prices, ResearchLeadLag(CHALLENGER_530), 249, 499)
    rayan = backtest(prices, ResearchLeadLag(RAYAN_BETA_531), 249, 499)

    rng = np.random.default_rng(20260714)
    noise_rows = []
    for bps in (1, 5, 10, 20):
        differences = []
        for _ in range(50):
            observed = prices * np.exp(rng.normal(0.0, bps * 1e-4, prices.shape))
            first = backtest(prices, ResearchLeadLag(CHALLENGER_530), 249, 499, observed_prices=observed)["score"]
            second = backtest(prices, ResearchLeadLag(RAYAN_BETA_531), 249, 499, observed_prices=observed)["score"]
            differences.append(second - first)
        differences = np.asarray(differences)
        noise_rows.append(
            {
                "noise_bps": bps,
                "difference_p05": float(np.quantile(differences, 0.05)),
                "difference_median": float(np.median(differences)),
                "difference_p95": float(np.quantile(differences, 0.95)),
                "rayan_win_fraction": float(np.mean(differences > 0.0)),
            }
        )
    noise = pd.DataFrame(noise_rows)
    noise.to_csv(TABLES / "rayan_beta_noise.csv", index=False)

    rng = np.random.default_rng(531)
    differences = []
    block = 20
    for _ in range(10_000):
        starts = rng.integers(0, 250 - block + 1, size=13)
        indices = np.concatenate([np.arange(start, start + block) for start in starts])[:250]
        first = reference["pnl"][indices]
        second = rayan["pnl"][indices]
        differences.append(
            competition_score(float(second.mean()), float(second.std()))
            - competition_score(float(first.mean()), float(first.std()))
        )
    differences = np.asarray(differences)

    long_prices = extended_prices(prices)
    model = ResearchLeadLag(RAYAN_BETA_531)
    started = time.perf_counter()
    for day in range(1_500, 1_750):
        model.getMyPosition(long_prices[:, :day])
    long_runtime = time.perf_counter() - started

    print(
        f"Challenger530 score={reference['score']:.2f} mean={reference['mean_pnl']:.2f} "
        f"std={reference['std_pnl']:.2f} sharpe={reference['sharpe']:.2f}"
    )
    print(
        f"RayanBeta531 score={rayan['score']:.2f} mean={rayan['mean_pnl']:.2f} "
        f"std={rayan['std_pnl']:.2f} sharpe={rayan['sharpe']:.2f}"
    )
    print(
        "bootstrap difference p05/median/p95/win",
        *[f"{value:.2f}" for value in np.quantile(differences, (0.05, 0.50, 0.95))],
        f"{np.mean(differences > 0):.1%}",
    )
    print("1750-day 250-call runtime", f"{long_runtime:.3f}s")
    print(noise.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    main()
