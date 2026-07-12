#!/usr/bin/env python3
"""Stress-test the multi-sleeve candidate without changing teamName.py."""

from __future__ import annotations

import importlib
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis" / "output" / "tables"
OUTPUT.mkdir(parents=True, exist_ok=True)


def score(mean_pnl: float, std_pnl: float) -> float:
    if mean_pnl <= 0 or std_pnl < 1e-10:
        return mean_pnl
    sharpe = np.sqrt(250) * mean_pnl / std_pnl
    return mean_pnl * sharpe**2 / (sharpe**2 + 1)


def backtest(prices, strategy, start, end, commission_multiplier=1.0):
    if hasattr(strategy, "resetState"):
        strategy.resetState()

    n_instruments = prices.shape[0]
    limits = np.full(n_instruments, 10_000.0)
    limits[0] = 100_000.0
    fees = np.full(n_instruments, 0.0001) * commission_multiplier
    fees[0] = 0.00002 * commission_multiplier
    position = np.zeros(n_instruments, dtype=int)
    pnl = []
    volume = 0.0

    started = time.perf_counter()
    for day in range(start, end):
        current_price = prices[:, day]
        requested = np.asarray(strategy.getMyPosition(prices[:, : day + 1]))
        max_shares = (limits / current_price).astype(int)
        new_position = np.clip(requested, -max_shares, max_shares).astype(int)
        trade = new_position - position
        traded_dollars = current_price * np.abs(trade)
        commission = float(np.sum(traded_dollars * fees))
        pnl.append(float(new_position @ (prices[:, day + 1] - current_price) - commission))
        volume += float(traded_dollars.sum())
        position = new_position

    elapsed = time.perf_counter() - started
    pnl = np.asarray(pnl)
    mean_pnl = float(pnl.mean())
    std_pnl = float(pnl.std())
    sharpe = float(np.sqrt(250) * mean_pnl / std_pnl) if std_pnl else 0.0
    return {
        "score": score(mean_pnl, std_pnl),
        "mean_pnl": mean_pnl,
        "std_pnl": std_pnl,
        "sharpe": sharpe,
        "volume": volume,
        "elapsed": elapsed,
        "pnl": pnl,
    }


def contract_checks(prices, candidate):
    candidate.resetState()
    limits = np.full(prices.shape[0], 10_000.0)
    limits[0] = 100_000.0
    for n_days in (1, 6, 119, 120, 121, 250, 500):
        output = candidate.getMyPosition(prices[:, :n_days])
        max_shares = (limits / prices[:, n_days - 1]).astype(int)
        assert output.shape == (51,)
        assert np.issubdtype(output.dtype, np.integer)
        assert np.all(np.abs(output) <= max_shares)
        assert np.array_equal(output, candidate.getMyPosition(prices[:, :n_days]))

    # A shorter history must reset state cleanly.
    reset_output = candidate.getMyPosition(prices[:, :200])
    max_shares = (limits / prices[:, 199]).astype(int)
    assert np.all(np.abs(reset_output) <= max_shares)

    # A large current-price shock must still obey the moving share caps.
    shocked = prices[:, :250].copy()
    shocked[:, -1] *= 10.0
    candidate.resetState()
    shocked_output = candidate.getMyPosition(shocked)
    shocked_limits = (limits / shocked[:, -1]).astype(int)
    assert np.all(np.abs(shocked_output) <= shocked_limits)


def set_config(candidate, original, changes):
    for key, value in original.items():
        setattr(candidate, key, value)
    for key, value in changes.items():
        setattr(candidate, key, value)
    candidate.resetState()


def main():
    prices = pd.read_csv(ROOT / "prices.txt", sep=r"\s+").values.T
    candidate = importlib.import_module("analysis.candidate_multisleeve")
    current = importlib.import_module("teamName")

    contract_checks(prices, candidate)
    print("Competition contract checks: PASS")

    configurable = (
        "REVERSAL_WEIGHTS",
        "REVERSAL_SLEEVE_WEIGHT",
        "PAIR_SLEEVE_WEIGHT",
        "SIGNAL_STRENGTH",
        "PAIR_ENTRY_Z",
        "PAIR_EXIT_Z",
        "DEADBAND_FRACTION",
        "HEDGE_FRACTION",
        "PAIRS",
    )
    original = {key: getattr(candidate, key) for key in configurable}

    tests = [("default", {})]
    tests += [
        (f"sleeves_{reversal:.1f}_{pair:.1f}", {"REVERSAL_SLEEVE_WEIGHT": reversal, "PAIR_SLEEVE_WEIGHT": pair})
        for reversal, pair in ((1.0, 0.0), (0.8, 0.2), (0.4, 0.6), (0.0, 1.0))
    ]
    tests += [(f"strength_{value}", {"SIGNAL_STRENGTH": value}) for value in (1.0, 3.0)]
    tests += [(f"entry_{value}", {"PAIR_ENTRY_Z": value}) for value in (1.0, 1.5, 2.0)]
    tests += [(f"deadband_{value}", {"DEADBAND_FRACTION": value}) for value in (0.0, 0.03, 0.05)]
    tests += [(f"hedge_{value}", {"HEDGE_FRACTION": value}) for value in (0.0, 0.5)]
    tests += [
        ("reversal_20_only", {"REVERSAL_WEIGHTS": (1.0, 0.0, 0.0)}),
        ("reversal_20_80", {"REVERSAL_WEIGHTS": (0.4, 0.6, 0.0)}),
        ("reversal_120_only", {"REVERSAL_WEIGHTS": (0.0, 0.0, 1.0)}),
        ("wrong_HUXZ_RCRI_index", {"PAIRS": ((1, 20), (8, 30))}),
    ]

    rows = []
    for name, changes in tests:
        set_config(candidate, original, changes)
        result = backtest(prices, candidate, 249, 499)
        rows.append({"test": name, **{k: v for k, v in result.items() if k != "pnl"}})

    set_config(candidate, original, {})
    current_result = backtest(prices, current, 249, 499)
    rows.append({"test": "current_teamName", **{k: v for k, v in current_result.items() if k != "pnl"}})
    sensitivity = pd.DataFrame(rows).set_index("test").sort_values("score", ascending=False)
    sensitivity.to_csv(OUTPUT / "multisleeve_sensitivity.csv")
    print("\nOfficial-window and one-at-a-time sensitivity")
    print(sensitivity.to_string(float_format=lambda value: f"{value:,.2f}"))

    set_config(candidate, original, {})
    folds = []
    for start in range(120, 470, 50):
        end = min(start + 50, 499)
        candidate_result = backtest(prices, candidate, start, end)
        current_fold = backtest(prices, current, start, end)
        folds.append(
            {
                "start": start + 1,
                "end": end + 1,
                "candidate_score": candidate_result["score"],
                "candidate_mean": candidate_result["mean_pnl"],
                "candidate_std": candidate_result["std_pnl"],
                "current_score": current_fold["score"],
                "current_mean": current_fold["mean_pnl"],
                "current_std": current_fold["std_pnl"],
            }
        )
    fold_frame = pd.DataFrame(folds)
    fold_frame.to_csv(OUTPUT / "multisleeve_walk_forward.csv", index=False)
    print("\nWalk-forward folds")
    print(fold_frame.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))

    cost_rows = []
    for multiplier in (1.0, 2.0, 5.0, 10.0):
        set_config(candidate, original, {})
        result = backtest(prices, candidate, 249, 499, multiplier)
        cost_rows.append({"commission_multiplier": multiplier, **{k: v for k, v in result.items() if k != "pnl"}})
    cost_frame = pd.DataFrame(cost_rows)
    cost_frame.to_csv(OUTPUT / "multisleeve_cost_stress.csv", index=False)
    print("\nCommission stress")
    print(cost_frame.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))

    # Moving-block bootstrap preserves short-run dependence better than IID resampling.
    set_config(candidate, original, {})
    base = backtest(prices, candidate, 249, 499)
    rng = np.random.default_rng(2026)
    block_length = 20
    bootstrap_scores = []
    for _ in range(5_000):
        starts = rng.integers(0, len(base["pnl"]) - block_length + 1, size=13)
        sample = np.concatenate([base["pnl"][start : start + block_length] for start in starts])[:250]
        bootstrap_scores.append(score(float(sample.mean()), float(sample.std())))
    quantiles = np.quantile(bootstrap_scores, (0.05, 0.50, 0.95))
    print(
        "\n20-day moving-block bootstrap score 5%/50%/95%:",
        ", ".join(f"{value:,.2f}" for value in quantiles),
        "positive probability:",
        f"{np.mean(np.asarray(bootstrap_scores) > 0):.1%}",
    )

    set_config(candidate, original, {})


if __name__ == "__main__":
    main()
