#!/usr/bin/env python3
"""Held-out calibration experiments around the residual lead-lag strategy.

This laboratory deliberately limits the adaptive layer to two components:
the proven one-day residual network and residual reversal.  Component weights
are estimated on a chronological validation block that was not used to fit
the validation lead-lag matrix.  This avoids selecting targets from their
in-sample correlation with the same observations used to estimate the graph.
"""

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


def cs(values, axis=None):
    """Standardize a vector or every column of a matrix."""
    if axis is None:
        scale = float(np.std(values))
        return np.zeros_like(values) if scale < 1e-12 else (values - np.mean(values)) / scale
    mean = np.mean(values, axis=axis, keepdims=True)
    scale = np.maximum(np.std(values, axis=axis, keepdims=True), 1e-12)
    return (values - mean) / scale


def lead_matrix(z, threshold_se=1.0):
    predictors = z[:, :-1]
    targets = z[:, 1:]
    count = predictors.shape[1]
    matrix = predictors @ targets.T / max(count, 1)
    np.fill_diagonal(matrix, 0.0)
    matrix[np.abs(matrix) < threshold_se / np.sqrt(max(count, 1))] = 0.0
    return matrix


def rolling_reversal(z, starts, lookback):
    """Return cross-sectionally standardized reversal at predictor indices."""
    cumulative = np.concatenate([np.zeros((z.shape[0], 1)), np.cumsum(z, axis=1)], axis=1)
    output = np.empty((z.shape[0], len(starts)))
    for column, end in enumerate(starts):
        begin = max(0, end + 1 - lookback)
        output[:, column] = -(cumulative[:, end + 1] - cumulative[:, begin])
    return cs(output, axis=0)


@dataclass(frozen=True)
class Config:
    calibration: str = "none"
    validation_window: int = 120
    calibration_strength: float = 0.50
    calibration_ridge: float = 25.0
    activation_history: int = 180
    reversal_weight: float = 0.20
    long_reversal_weight: float = 0.075
    hysteresis: float = 0.30
    hedge_fraction: float = 1.0
    hedge_beta_window: int = 120


class CalibratedLeadLag:
    def __init__(self, config=Config()):
        self.config = config
        self.resetState()

    def resetState(self):
        self.previous_direction = None
        self.last_day_count = None
        self.last_position = None

    def _validation_components(self, z):
        total = z.shape[1]
        split = max(60, total - self.config.validation_window)
        if split < 60 or total - split < 30:
            return None

        validation_matrix = lead_matrix(z[:, :split])
        predictor_indices = np.arange(split - 1, total - 1)
        lead = cs(validation_matrix.T @ z[:, predictor_indices], axis=0)
        fast = rolling_reversal(z, predictor_indices, 20)
        slow = rolling_reversal(z, predictor_indices, 60)
        reversal = cs(
            (1.0 - self.config.long_reversal_weight) * fast
            + self.config.long_reversal_weight * slow,
            axis=0,
        )
        outcome = cs(z[:, split:], axis=0)
        return lead, reversal, outcome

    def _calibrate(self, lead_now, reversal_now, z):
        if self.config.calibration == "none" or z.shape[1] < self.config.activation_history:
            return (1.0 - self.config.reversal_weight) * lead_now + self.config.reversal_weight * reversal_now

        validation = self._validation_components(z)
        if validation is None:
            return (1.0 - self.config.reversal_weight) * lead_now + self.config.reversal_weight * reversal_now
        lead, reversal, outcome = validation
        strength = self.config.calibration_strength

        if self.config.calibration == "global_stack":
            design = np.stack([lead.ravel(), reversal.ravel()], axis=1)
            target = outcome.ravel()
            gram = design.T @ design + self.config.calibration_ridge * np.eye(2)
            weights = np.maximum(np.linalg.solve(gram, design.T @ target), 0.0)
            if float(weights.sum()) < 1e-12:
                learned = np.array([1.0 - self.config.reversal_weight, self.config.reversal_weight])
            else:
                learned = weights / weights.sum()
            fixed = np.array([1.0 - self.config.reversal_weight, self.config.reversal_weight])
            weights = (1.0 - strength) * fixed + strength * learned
            return weights[0] * lead_now + weights[1] * reversal_now

        lead_centered = lead - lead.mean(axis=1, keepdims=True)
        outcome_centered = outcome - outcome.mean(axis=1, keepdims=True)
        reliability = np.sum(lead_centered * outcome_centered, axis=1) / np.sqrt(
            np.maximum(np.sum(lead_centered * lead_centered, axis=1), 1e-12)
            * np.maximum(np.sum(outcome_centered * outcome_centered, axis=1), 1e-12)
        )

        if self.config.calibration == "target_gate":
            multiplier = np.where(reliability > 0.0, 1.0, 0.0)
        elif self.config.calibration == "target_scale":
            ranks = np.argsort(np.argsort(reliability)).astype(float) / (len(reliability) - 1)
            multiplier = (1.0 - strength) + 2.0 * strength * ranks
        elif self.config.calibration == "target_signed":
            multiplier = (1.0 - strength) + strength * np.sign(reliability)
        else:
            raise ValueError(self.config.calibration)

        calibrated_lead = cs(multiplier * lead_now)
        return (1.0 - self.config.reversal_weight) * calibrated_lead + self.config.reversal_weight * reversal_now

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

        residual = returns[1:] - returns[0]
        mean = residual.mean(axis=1, keepdims=True)
        scale = np.maximum(residual.std(axis=1, keepdims=True), 1e-12)
        z = (residual - mean) / scale

        matrix = lead_matrix(z)
        lead_now = cs(matrix.T @ z[:, -1])
        fast = cs(-z[:, -20:].sum(axis=1))
        slow = cs(-z[:, -60:].sum(axis=1))
        reversal_now = cs(
            (1.0 - self.config.long_reversal_weight) * fast
            + self.config.long_reversal_weight * slow
        )
        signal = self._calibrate(lead_now, reversal_now, z)
        signal -= signal.mean()

        direction = np.sign(signal)
        if self.previous_direction is not None:
            weak = np.abs(signal) < self.config.hysteresis * np.mean(np.abs(signal))
            direction = np.where(weak, self.previous_direction, direction)
        self.previous_direction = direction.copy()

        current = prices[:, -1]
        limits = (ASSET_CAP / current[1:]).astype(int)
        positions[1:] = direction.astype(int) * limits

        sample = returns[:, -self.config.hedge_beta_window :]
        factor = sample[0] - sample[0].mean()
        betas = (
            (sample[1:] - sample[1:].mean(axis=1, keepdims=True)) @ factor
        ) / max(float(factor @ factor), 1e-12)
        factor_dollars = float((positions[1:] * current[1:]) @ betas)
        hedge = np.clip(-self.config.hedge_fraction * factor_dollars, -0.999 * ALGO_CAP, 0.999 * ALGO_CAP)
        algo_limit = int(ALGO_CAP / current[0])
        positions[0] = int(np.clip(np.trunc(hedge / current[0]), -algo_limit, algo_limit))

        self.last_day_count = n_days
        self.last_position = positions.astype(int)
        return positions.astype(int).copy()


def evaluate(prices, label, config):
    strategy = CalibratedLeadLag(config)
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
        "last100": backtest(prices, strategy, 399, 499)["score"],
        "cost10": backtest(prices, strategy, 249, 499, 10.0)["score"],
        "config": repr(config),
    }


def variants():
    base = Config()
    yield "challenger_equivalent", base
    for mode in ("global_stack", "target_gate", "target_scale", "target_signed"):
        for window in (80, 120, 180):
            for strength in (0.25, 0.50, 0.75, 1.00):
                yield (
                    f"{mode}_v{window}_s{strength}",
                    replace(base, calibration=mode, validation_window=window, calibration_strength=strength),
                )


def main():
    prices = pd.read_csv(ROOT / "prices.txt", sep=r"\s+").values.T
    results = pd.DataFrame([evaluate(prices, label, config) for label, config in variants()])
    output = ROOT / "analysis" / "output" / "tables" / "top10_calibration_lab.csv"
    results.to_csv(output, index=False)
    columns = [
        "strategy", "score", "mean", "std", "sharpe", "early", "fold_median",
        "fold_min", "positive_folds", "window_median", "window_min", "last100", "cost10",
    ]
    print(results.sort_values("score", ascending=False)[columns].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    main()
