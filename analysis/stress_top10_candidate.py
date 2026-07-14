#!/usr/bin/env python3
"""Paired robustness tests for the mature beta-shrinkage challenger."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.beta_shrink_top10_lab import BetaShrinkLeadLag, Config
from analysis.leadlag_improvement_lab import Config as OldConfig
from analysis.leadlag_improvement_lab import ResearchLeadLag
from analysis.stress_final_strategy import backtest, competition_score


TABLES = ROOT / "analysis" / "output" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def load_challenger():
    path = ROOT / "analysis" / "output" / "challenger" / "negative alpha.py"
    spec = importlib.util.spec_from_file_location("challenger_530_stress", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def summary(prices, label, strategy):
    official = backtest(prices, strategy, 249, 499)
    early = backtest(prices, strategy, 100, 249)
    folds = [
        backtest(prices, strategy, start, min(start + 50, 499))["score"]
        for start in range(100, 499, 50)
    ]
    windows = [
        backtest(prices, strategy, start, min(start + 250, 499))["score"]
        for start in (60, 100, 150, 200, 249)
    ]
    cost = backtest(prices, strategy, 249, 499, 10.0)
    return {
        "strategy": label,
        "score": official["score"],
        "mean": official["mean_pnl"],
        "std": official["std_pnl"],
        "sharpe": official["sharpe"],
        "volume": official["volume"],
        "early": early["score"],
        "fold_median": float(np.median(folds)),
        "fold_min": float(np.min(folds)),
        "positive_folds": int(np.sum(np.asarray(folds) > 0.0)),
        "window_median": float(np.median(windows)),
        "window_min": float(np.min(windows)),
        "last200": backtest(prices, strategy, 299, 499)["score"],
        "last150": backtest(prices, strategy, 349, 499)["score"],
        "last100": backtest(prices, strategy, 399, 499)["score"],
        "cost10": cost["score"],
        "pnl": official["pnl"],
    }


def paired_bootstrap(left, right, simulations=20_000, block=20):
    rng = np.random.default_rng(20260714)
    differences = np.empty(simulations)
    count = len(left)
    blocks = int(np.ceil(count / block))
    for simulation in range(simulations):
        starts = rng.integers(0, count - block + 1, size=blocks)
        indices = np.concatenate([np.arange(start, start + block) for start in starts])[:count]
        left_sample = left[indices]
        right_sample = right[indices]
        left_score = competition_score(float(left_sample.mean()), float(left_sample.std()))
        right_score = competition_score(float(right_sample.mean()), float(right_sample.std()))
        differences[simulation] = right_score - left_score
    return {
        "p05": float(np.quantile(differences, 0.05)),
        "median": float(np.median(differences)),
        "p95": float(np.quantile(differences, 0.95)),
        "win_rate": float(np.mean(differences > 0.0)),
    }


def noise_trials(prices, candidate_config, simulations=50):
    rng = np.random.default_rng(260714)
    rows = []
    for bps in (1, 5, 10, 20):
        differences = []
        challenger_scores = []
        candidate_scores = []
        for _ in range(simulations):
            observed = prices * np.exp(rng.normal(0.0, bps * 1e-4, prices.shape))
            challenger_result = backtest(
                prices, load_challenger(), 249, 499, observed_prices=observed
            )
            candidate_result = backtest(
                prices,
                BetaShrinkLeadLag(candidate_config),
                249,
                499,
                observed_prices=observed,
            )
            challenger_scores.append(challenger_result["score"])
            candidate_scores.append(candidate_result["score"])
            differences.append(candidate_result["score"] - challenger_result["score"])
        rows.append(
            {
                "noise_bps": bps,
                "challenger_median": float(np.median(challenger_scores)),
                "candidate_median": float(np.median(candidate_scores)),
                "difference_p05": float(np.quantile(differences, 0.05)),
                "difference_median": float(np.median(differences)),
                "difference_p95": float(np.quantile(differences, 0.95)),
                "candidate_win_rate": float(np.mean(np.asarray(differences) > 0.0)),
            }
        )
    return pd.DataFrame(rows)


def direction_difference(prices, candidate_config):
    challenger = load_challenger()
    candidate = BetaShrinkLeadLag(candidate_config)
    challenger.resetState()
    candidate.resetState()
    differences = []
    for day in range(249, 499):
        old = challenger.getMyPosition(prices[:, : day + 1])
        new = candidate.getMyPosition(prices[:, : day + 1])
        differences.append(np.sum(np.sign(old[1:]) != np.sign(new[1:])))
    return np.asarray(differences)


def runtime_1750(prices, candidate_config):
    returns = np.diff(np.log(prices), axis=1)
    needed = 1749 - returns.shape[1]
    tiled = np.tile(returns, int(np.ceil(needed / returns.shape[1])))[:, :needed]
    extended_returns = np.concatenate([returns, tiled], axis=1)
    extended = np.empty((prices.shape[0], 1750))
    extended[:, 0] = prices[:, 0]
    extended[:, 1:] = prices[:, [0]] * np.exp(np.cumsum(extended_returns, axis=1))
    strategy = BetaShrinkLeadLag(candidate_config)
    started = time.perf_counter()
    for day in range(1500, 1750):
        strategy.getMyPosition(extended[:, : day + 1])
    return time.perf_counter() - started


def main():
    prices = pd.read_csv(ROOT / "prices.txt", sep=r"\s+").values.T
    candidate_config = Config(
        alpha_beta_shrinkage=0.25,
        ramp_start=200,
        ramp_end=300,
        long_reversal_weight=0.10,
    )
    challenger = summary(prices, "Challenger 530", load_challenger())
    candidate = summary(
        prices,
        "Mature Beta Challenger",
        BetaShrinkLeadLag(candidate_config),
    )
    comparison = pd.DataFrame(
        [{key: value for key, value in row.items() if key != "pnl"} for row in (challenger, candidate)]
    )
    comparison.to_csv(TABLES / "top10_candidate_comparison.csv", index=False)

    bootstrap = paired_bootstrap(challenger["pnl"], candidate["pnl"])
    noise = noise_trials(prices, candidate_config)
    noise.to_csv(TABLES / "top10_candidate_noise.csv", index=False)

    frozen_candidate = backtest(
        prices,
        BetaShrinkLeadLag(replace(candidate_config, freeze_at=249)),
        249,
        499,
    )
    frozen_old_config = OldConfig(
        long_reversal_lookback=60,
        long_reversal_weight=0.075,
        hedge_beta_window=120,
        hedge_beta_shrinkage=1.0,
        freeze_at=249,
    )
    frozen_challenger = backtest(
        prices,
        ResearchLeadLag(frozen_old_config),
        249,
        499,
    )
    differences = direction_difference(prices, candidate_config)
    runtime = runtime_1750(prices, candidate_config)

    print(comparison.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print("\nPaired moving-block bootstrap", bootstrap)
    print("\nObservation-noise stress\n", noise.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print(
        "\nFrozen scores challenger/candidate",
        f"{frozen_challenger['score']:.2f}",
        f"{frozen_candidate['score']:.2f}",
    )
    print(
        "Direction differences median/mean/max",
        f"{np.median(differences):.1f}",
        f"{np.mean(differences):.1f}",
        int(np.max(differences)),
    )
    print("Runtime 250 calls through day 1750", f"{runtime:.3f}s")


if __name__ == "__main__":
    main()
