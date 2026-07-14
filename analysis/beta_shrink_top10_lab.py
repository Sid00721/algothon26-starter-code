#!/usr/bin/env python3
"""Mature-sample beta-shrinkage residual experiments for a top-10 challenger."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.stress_final_strategy import backtest


ASSET_CAP = 10_000.0
ALGO_CAP = 100_000.0


def cs(values):
    scale = float(np.std(values))
    return np.zeros_like(values) if scale < 1e-12 else (values - np.mean(values)) / scale


@dataclass(frozen=True)
class Config:
    alpha_beta_shrinkage: float = 0.25
    ramp_start: int = 200
    ramp_end: int = 300
    significance_se: float = 1.0
    reversal_weight: float = 0.20
    long_reversal_weight: float = 0.075
    hysteresis: float = 0.30
    hedge_beta_window: int = 120
    hedge_fraction: float = 1.0
    freeze_at: int = 0


class BetaShrinkLeadLag:
    def __init__(self, config=Config()):
        self.config = config
        self.resetState()

    def resetState(self):
        self.previous_direction = None
        self.last_day_count = None
        self.last_position = None

    def getMyPosition(self, prcSoFar):
        prices = np.asarray(prcSoFar, dtype=float)
        n_instruments, n_days = prices.shape
        if self.last_day_count == n_days and self.last_position is not None:
            return self.last_position.copy()
        if self.last_day_count is not None and n_days < self.last_day_count:
            self.resetState()

        positions = np.zeros(n_instruments, dtype=int)
        returns = np.diff(np.log(prices), axis=1)
        if n_instruments != 51 or returns.shape[1] < 60:
            self.last_day_count = n_days
            self.last_position = positions
            return positions.copy()

        if self.config.freeze_at > 0 and returns.shape[1] > self.config.freeze_at:
            beta_fit = returns[:, : self.config.freeze_at]
        else:
            beta_fit = returns
        factor = beta_fit[0] - beta_fit[0].mean()
        denominator = max(float(factor @ factor), 1e-12)
        raw_betas = (
            (beta_fit[1:] - beta_fit[1:].mean(axis=1, keepdims=True)) @ factor
        ) / denominator
        if self.config.ramp_end <= self.config.ramp_start:
            maturity = 1.0
        else:
            maturity = float(
                np.clip(
                    (beta_fit.shape[1] - self.config.ramp_start)
                    / (self.config.ramp_end - self.config.ramp_start),
                    0.0,
                    1.0,
                )
            )
        alpha_betas = 1.0 + maturity * self.config.alpha_beta_shrinkage * (raw_betas - 1.0)

        residual = returns[1:] - alpha_betas[:, None] * returns[0]
        if self.config.freeze_at > 0 and residual.shape[1] > self.config.freeze_at:
            estimation = residual[:, : self.config.freeze_at]
        else:
            estimation = residual
        mean = estimation.mean(axis=1, keepdims=True)
        scale = np.maximum(estimation.std(axis=1, keepdims=True), 1e-12)
        z_estimation = (estimation - mean) / scale
        z = (residual - mean) / scale
        predictors = z_estimation[:, :-1]
        targets = z_estimation[:, 1:]
        count = predictors.shape[1]
        matrix = predictors @ targets.T / count
        np.fill_diagonal(matrix, 0.0)
        matrix[np.abs(matrix) < self.config.significance_se / np.sqrt(count)] = 0.0

        lead = cs(matrix.T @ z[:, -1])
        fast = cs(-z[:, -20:].sum(axis=1))
        slow = cs(-z[:, -60:].sum(axis=1))
        reversal = cs(
            (1.0 - self.config.long_reversal_weight) * fast
            + self.config.long_reversal_weight * slow
        )
        signal = (1.0 - self.config.reversal_weight) * lead + self.config.reversal_weight * reversal
        signal -= signal.mean()

        direction = np.sign(signal)
        if self.previous_direction is not None:
            weak = np.abs(signal) < self.config.hysteresis * np.mean(np.abs(signal))
            direction = np.where(weak, self.previous_direction, direction)
        self.previous_direction = direction.copy()

        current = prices[:, -1]
        limits = (ASSET_CAP / current[1:]).astype(int)
        positions[1:] = direction.astype(int) * limits

        hedge_sample = returns[:, -self.config.hedge_beta_window :]
        hedge_factor = hedge_sample[0] - hedge_sample[0].mean()
        hedge_denominator = max(float(hedge_factor @ hedge_factor), 1e-12)
        hedge_betas = (
            (hedge_sample[1:] - hedge_sample[1:].mean(axis=1, keepdims=True)) @ hedge_factor
        ) / hedge_denominator
        factor_dollars = float((positions[1:] * current[1:]) @ hedge_betas)
        hedge_dollars = np.clip(
            -self.config.hedge_fraction * factor_dollars,
            -0.999 * ALGO_CAP,
            0.999 * ALGO_CAP,
        )
        algo_limit = int(ALGO_CAP / current[0])
        positions[0] = int(
            np.clip(np.trunc(hedge_dollars / current[0]), -algo_limit, algo_limit)
        )

        self.last_day_count = n_days
        self.last_position = positions.astype(int)
        return positions.astype(int).copy()


def evaluate(prices, label, config):
    strategy = BetaShrinkLeadLag(config)
    official = backtest(prices, strategy, 249, 499)
    early = backtest(prices, strategy, 100, 249)
    folds = [
        backtest(prices, strategy, start, min(start + 50, 499))["score"]
        for start in range(100, 499, 50)
    ]
    windows = [
        backtest(prices, strategy, start, start + 250)["score"]
        for start in range(0, 250, 50)
    ]
    return {
        "strategy": label,
        "score": official["score"],
        "mean": official["mean_pnl"],
        "std": official["std_pnl"],
        "sharpe": official["sharpe"],
        "early": early["score"],
        "fold_median": float(np.median(folds)),
        "fold_min": float(np.min(folds)),
        "positive_folds": int(np.sum(np.asarray(folds) > 0.0)),
        "window_median": float(np.median(windows)),
        "window_min": float(np.min(windows)),
        "last200": backtest(prices, strategy, 299, 499)["score"],
        "last150": backtest(prices, strategy, 349, 499)["score"],
        "last100": backtest(prices, strategy, 399, 499)["score"],
        "cost10": backtest(prices, strategy, 249, 499, 10.0)["score"],
        "config": repr(config),
    }


def variants():
    base = Config()
    yield "candidate", base
    for shrinkage in (0.15, 0.20, 0.225, 0.25, 0.275, 0.30, 0.35):
        yield f"shrink_{shrinkage}", replace(base, alpha_beta_shrinkage=shrinkage)
    for start in (150, 180, 200, 220, 250):
        for end in (280, 300, 320, 350):
            if start < end:
                yield f"ramp_{start}_{end}", replace(base, ramp_start=start, ramp_end=end)
    for long_weight in (0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15):
        yield f"long_{long_weight}", replace(base, long_reversal_weight=long_weight)
    for reversal_weight in (0.15, 0.175, 0.20, 0.225, 0.25):
        yield f"reversal_{reversal_weight}", replace(base, reversal_weight=reversal_weight)


def main():
    prices = pd.read_csv(ROOT / "prices.txt", sep=r"\s+").values.T
    results = pd.DataFrame([evaluate(prices, label, config) for label, config in variants()])
    output = ROOT / "analysis" / "output" / "tables" / "beta_shrink_top10_lab.csv"
    results.to_csv(output, index=False)
    columns = [
        "strategy", "score", "mean", "std", "sharpe", "early", "fold_median",
        "fold_min", "positive_folds", "window_median", "window_min", "last200",
        "last150", "last100", "cost10",
    ]
    print(results.sort_values("score", ascending=False)[columns].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    main()
