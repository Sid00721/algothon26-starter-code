#!/usr/bin/env python3
"""Robust comparison of the saved lead-lag model and index-inspired variants.

This script deliberately separates model selection from the submission file.
It tests the broad design decisions suggested by the index-arbitrage prompt:
explicit ALGO hedging, residual/PCA factor removal, and small model ensembles.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.index_strategy_lab import PCAResidualLeadLag
from analysis.stress_final_strategy import backtest


TABLES = ROOT / "analysis" / "output" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


class LeadLagVariant:
    """Adapter exposing one fixed configuration of candidate_leadlag."""

    def __init__(
        self,
        reversal_weight=0.20,
        hedge_fraction=1.0,
        threshold_se=1.0,
        hysteresis=0.30,
        freeze_at=0,
    ):
        self.module = importlib.import_module("analysis.candidate_leadlag")
        self.reversal_weight = reversal_weight
        self.hedge_fraction = hedge_fraction
        self.threshold_se = threshold_se
        self.hysteresis = hysteresis
        self.freeze_at = freeze_at

    def _configure(self):
        self.module.REVERSAL_WEIGHT = self.reversal_weight
        self.module.HEDGE_FRACTION = self.hedge_fraction
        self.module.SIGNIFICANCE_SE = self.threshold_se
        self.module.HYSTERESIS_FRACTION = self.hysteresis
        self.module.FREEZE_ESTIMATION_AT = self.freeze_at
        self.module.ESTIMATION_WINDOW = 0
        self.module.SIZING_MODE = "sign"

    def resetState(self):
        self._configure()
        self.module.resetState()

    def getMyPosition(self, prices):
        self._configure()
        return self.module.getMyPosition(prices)


class PositionBlend:
    """Convex position blend; PCA receives no separate ALGO hedge."""

    def __init__(self, pca_weight):
        self.pca_weight = pca_weight
        self.leadlag = LeadLagVariant(reversal_weight=0.20, hedge_fraction=0.0)
        self.pca = PCAResidualLeadLag(factors=2)

    def resetState(self):
        self.leadlag.resetState()
        self.pca.resetState()

    def getMyPosition(self, prices):
        leadlag_position = self.leadlag.getMyPosition(prices).astype(float)
        pca_position = self.pca.getMyPosition(prices).astype(float)
        pca_position[0] = 0.0
        target = (1.0 - self.pca_weight) * leadlag_position + self.pca_weight * pca_position
        return np.rint(target).astype(int)


def evaluate(prices, label, strategy, cost_multiplier=10.0):
    early = backtest(prices, strategy, 100, 249)
    official = backtest(prices, strategy, 249, 499)
    folds = [backtest(prices, strategy, start, min(start + 50, 499))["score"] for start in range(100, 499, 50)]
    high_cost = backtest(prices, strategy, 249, 499, commission_multiplier=cost_multiplier)
    return {
        "strategy": label,
        "official_score": official["score"],
        "official_mean": official["mean_pnl"],
        "official_std": official["std_pnl"],
        "early_score": early["score"],
        "fold_median": float(np.median(folds)),
        "fold_min": float(np.min(folds)),
        "positive_folds": int(np.sum(np.asarray(folds) > 0)),
        "cost_10x_score": high_cost["score"],
        "volume": official["volume"],
    }


def main():
    prices = pd.read_csv(ROOT / "prices.txt", sep=r"\s+").values.T
    rows = []

    # Broad two-dimensional design grid.  The candidate is not chosen merely
    # by the maximum visible score; temporal stability is retained beside it.
    for reversal in (0.0, 0.10, 0.15, 0.20, 0.25):
        for hedge in (0.0, 0.25, 0.50, 0.75, 1.0):
            label = f"leadlag_r{reversal:.2f}_h{hedge:.2f}"
            rows.append(evaluate(prices, label, LeadLagVariant(reversal, hedge)))

    for weight in (0.10, 0.20, 0.30, 0.50):
        rows.append(evaluate(prices, f"blend_PCA2_{weight:.2f}", PositionBlend(weight)))

    results = pd.DataFrame(rows)
    results.to_csv(TABLES / "index_candidate_stress.csv", index=False)

    # A strict pre-test freeze probes whether updating the lead-lag matrix is
    # genuinely useful, while avoiding a look-ahead interpretation.
    frozen_rows = []
    for reversal in (0.0, 0.15, 0.20):
        for hedge in (0.0, 0.50, 1.0):
            strategy = LeadLagVariant(reversal, hedge, freeze_at=249)
            result = backtest(prices, strategy, 249, 499)
            frozen_rows.append(
                {
                    "reversal_weight": reversal,
                    "hedge_fraction": hedge,
                    **{key: result[key] for key in ("score", "mean_pnl", "std_pnl", "sharpe", "volume")},
                }
            )
    frozen = pd.DataFrame(frozen_rows)
    frozen.to_csv(TABLES / "index_candidate_frozen.csv", index=False)

    print(results.sort_values("official_score", ascending=False).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print("\nFrozen through day 249\n", frozen.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    main()
