#!/usr/bin/env python3
"""Contrarian audit of Rayan's proposed structural refinements.

The validated Challenger 530 configuration is the reference.  Variants test
volatility-scaled hysteresis, beta-window sensitivity, and an actual
multi-response L1 lead-lag estimator without changing unrelated components.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskLasso


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.leadlag_improvement_lab import Config, ResearchLeadLag
from analysis.stress_final_strategy import backtest


TABLES = ROOT / "analysis" / "output" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

CHALLENGER_530 = replace(
    Config(),
    long_reversal_lookback=60,
    long_reversal_weight=0.075,
    hedge_beta_window=120,
    hedge_beta_shrinkage=1.0,
)


class MultiTaskLassoLeadLag(ResearchLeadLag):
    """Group-L1 multi-response regression used as a sparse VAR(1)."""

    def __init__(self, config, alpha):
        self.alpha = alpha
        super().__init__(config)

    def _lead_matrix(self, z, lag):
        x = z[:, :-lag].T
        y = z[:, lag:].T
        model = MultiTaskLasso(
            alpha=self.alpha,
            fit_intercept=False,
            max_iter=2_000,
            tol=1e-4,
            selection="cyclic",
        )
        model.fit(x, y)
        matrix = model.coef_.T
        np.fill_diagonal(matrix, 0.0)
        return matrix


def evaluate(prices, label, factory):
    official = backtest(prices, factory(), 249, 499)
    early = backtest(prices, factory(), 100, 249)
    folds = [backtest(prices, factory(), start, min(start + 50, 499))["score"] for start in range(100, 499, 50)]
    windows = [backtest(prices, factory(), start, min(start + 250, 499))["score"] for start in (60, 100, 150, 200, 249)]
    cost = backtest(prices, factory(), 249, 499, commission_multiplier=10.0)
    return {
        "strategy": label,
        "official_score": official["score"],
        "official_mean": official["mean_pnl"],
        "official_std": official["std_pnl"],
        "official_sharpe": official["sharpe"],
        "early_score": early["score"],
        "fold_median": float(np.median(folds)),
        "fold_min": float(np.min(folds)),
        "positive_folds": int(np.sum(np.asarray(folds) > 0.0)),
        "window_median": float(np.median(windows)),
        "window_min": float(np.min(windows)),
        "cost_10x_score": cost["score"],
        "volume": official["volume"],
    }


def main():
    prices = pd.read_csv(ROOT / "prices.txt", sep=r"\s+").values.T
    variants = [("Challenger_530", lambda: ResearchLeadLag(CHALLENGER_530))]

    for window in (40, 60, 90, 120, 180, 250):
        for shrinkage in (0.50, 0.75, 1.00):
            config = replace(CHALLENGER_530, hedge_beta_window=window, hedge_beta_shrinkage=shrinkage)
            variants.append((f"beta_w{window}_s{shrinkage}", lambda config=config: ResearchLeadLag(config)))

    for power in (0.50, 1.00, 1.50):
        for hysteresis in (0.20, 0.25, 0.30, 0.35):
            config = replace(
                CHALLENGER_530,
                volatility_hysteresis=True,
                volatility_hysteresis_power=power,
                hysteresis=hysteresis,
            )
            variants.append((f"volhyst_h{hysteresis}_p{power}", lambda config=config: ResearchLeadLag(config)))

    # L1 is opt-in because the full causal grid takes minutes and weak-alpha
    # fits emit convergence warnings on these collinear predictors.  This is
    # research-only and is not a viable submission path.
    if "--include-l1" in sys.argv:
        for alpha in (0.010, 0.030, 0.100):
            variants.append((f"multitask_l1_{alpha}", lambda alpha=alpha: MultiTaskLassoLeadLag(CHALLENGER_530, alpha)))

    results = pd.DataFrame([evaluate(prices, label, factory) for label, factory in variants])
    results.to_csv(TABLES / "rayan_structural_lab.csv", index=False)
    print(results.sort_values("official_score", ascending=False).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    main()
