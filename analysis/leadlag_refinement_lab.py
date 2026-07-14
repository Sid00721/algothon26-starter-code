#!/usr/bin/env python3
"""Second-stage refinements around the robust live lead-lag baseline."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.leadlag_improvement_lab import Config, TABLES, result_row


def variants():
    base = Config()
    candidates = [("baseline", base)]
    candidates += [
        (f"hedge_{fraction:.3f}", replace(base, hedge_fraction=fraction))
        for fraction in (0.80, 0.85, 0.875, 0.90, 0.95)
    ]
    candidates += [
        (f"long60_{weight:.2f}", replace(base, long_reversal_lookback=60, long_reversal_weight=weight))
        for weight in (0.05, 0.10, 0.15, 0.20, 0.25)
    ]
    candidates += [
        (
            f"long60_{weight:.2f}_hedge0875",
            replace(base, long_reversal_lookback=60, long_reversal_weight=weight, hedge_fraction=0.875),
        )
        for weight in (0.05, 0.10, 0.15, 0.20)
    ]
    candidates += [
        (
            f"betahedge_w{window}_s{shrinkage}",
            replace(base, hedge_beta_window=window, hedge_beta_shrinkage=shrinkage),
        )
        for window in (0, 120, 250)
        for shrinkage in (0.25, 0.50, 1.00)
    ]
    candidates += [
        (
            f"long60_{long_weight:.3f}_beta120_s{shrinkage}",
            replace(
                base,
                long_reversal_lookback=60,
                long_reversal_weight=long_weight,
                hedge_beta_window=120,
                hedge_beta_shrinkage=shrinkage,
            ),
        )
        for long_weight in (0.025, 0.05, 0.075, 0.10, 0.125)
        for shrinkage in (0.50, 0.75, 1.00)
    ]
    return candidates


def main():
    prices = pd.read_csv(ROOT / "prices.txt", sep=r"\s+").values.T
    results = pd.DataFrame([result_row(prices, label, config) for label, config in variants()])
    results.to_csv(TABLES / "leadlag_refinement_lab.csv", index=False)
    columns = [
        "strategy",
        "official_score",
        "official_mean",
        "official_std",
        "early_score",
        "fold_median",
        "fold_min",
        "positive_folds",
        "window_median",
        "window_min",
        "frozen_score",
        "cost_10x_score",
    ]
    print(results.sort_values("official_score", ascending=False)[columns].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    main()
