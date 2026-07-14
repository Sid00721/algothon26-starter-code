#!/usr/bin/env python3
"""Causal improvement laboratory for the live residual lead-lag strategy.

The objective is not to maximize one public 250-day score.  Every variant is
measured on early data, eight chronological folds, five 250-day pseudo-hidden
windows, a strict frozen-network test, and 10x commissions.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.stress_final_strategy import backtest


TABLES = ROOT / "analysis" / "output" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

ASSET_CAP = 10_000.0
ALGO_CAP = 100_000.0


@dataclass(frozen=True)
class Config:
    lag_weights: tuple[tuple[int, float], ...] = ((1, 1.0),)
    significance_se: float = 1.0
    reversal_weight: float = 0.20
    reversal_lookback: int = 20
    reversal_mode: str = "linear"
    hysteresis: float = 0.30
    volatility_hysteresis: bool = False
    volatility_hysteresis_lookback: int = 20
    volatility_hysteresis_power: float = 1.0
    hedge_fraction: float = 1.0
    hedge_beta_window: int = 0
    hedge_beta_shrinkage: float = 0.0
    cross_sectional_demean: bool = False
    soft_threshold: bool = False
    recent_window: int = 0
    recent_weight: float = 0.0
    consensus_window: int = 0
    split_consensus: bool = False
    incoming_edges: int = 0
    asymmetry_mode: str = "none"
    svd_rank: int = 0
    column_normalization: str = "none"
    top_k: int = 50
    long_reversal_lookback: int = 0
    long_reversal_weight: float = 0.0
    freeze_at: int = 0


def standardize_cross_section(values):
    scale = values.std()
    if scale < 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / scale


class ResearchLeadLag:
    """Configurable, stateful, one-file-compatible lead-lag model."""

    def __init__(self, config=Config()):
        self.config = config
        self.resetState()

    def resetState(self):
        self.previous_direction = None
        self.last_day_count = None
        self.last_position = None

    def _raw_matrix(self, z, lag):
        predictors = z[:, :-lag]
        targets = z[:, lag:]
        count = predictors.shape[1]
        matrix = predictors @ targets.T / max(count, 1)
        np.fill_diagonal(matrix, 0.0)
        return matrix, count

    def _threshold(self, matrix, count):
        floor = self.config.significance_se / np.sqrt(max(count, 1))
        if self.config.soft_threshold:
            return np.sign(matrix) * np.maximum(np.abs(matrix) - floor, 0.0)
        return np.where(np.abs(matrix) >= floor, matrix, 0.0)

    def _lead_matrix(self, z, lag):
        full_raw, full_count = self._raw_matrix(z, lag)
        matrix = full_raw

        if self.config.recent_window > lag and z.shape[1] > self.config.recent_window:
            recent = z[:, -self.config.recent_window :]
            recent_raw, _ = self._raw_matrix(recent, lag)
            weight = self.config.recent_weight
            matrix = (1.0 - weight) * full_raw + weight * recent_raw

        matrix = self._threshold(matrix, full_count)

        if self.config.consensus_window > lag and z.shape[1] > self.config.consensus_window:
            recent = z[:, -self.config.consensus_window :]
            recent_raw, recent_count = self._raw_matrix(recent, lag)
            recent_matrix = self._threshold(recent_raw, recent_count)
            stable = (matrix != 0.0) & (recent_matrix != 0.0) & (np.sign(matrix) == np.sign(recent_matrix))
            matrix = np.where(stable, matrix, 0.0)

        if self.config.split_consensus and z.shape[1] >= max(120, 4 * lag):
            midpoint = z.shape[1] // 2
            first_raw, first_count = self._raw_matrix(z[:, :midpoint], lag)
            second_raw, second_count = self._raw_matrix(z[:, midpoint:], lag)
            first = self._threshold(first_raw, first_count)
            second = self._threshold(second_raw, second_count)
            stable = (first != 0.0) & (second != 0.0) & (np.sign(first) == np.sign(second))
            matrix = np.where(stable, matrix, 0.0)

        if self.config.asymmetry_mode == "difference":
            matrix = matrix - matrix.T
        elif self.config.asymmetry_mode == "dominant":
            matrix = np.where(np.abs(matrix) > np.abs(matrix.T), matrix, 0.0)
        elif self.config.asymmetry_mode == "excess":
            matrix = np.sign(matrix) * np.maximum(np.abs(matrix) - np.abs(matrix.T), 0.0)
        elif self.config.asymmetry_mode != "none":
            raise ValueError(self.config.asymmetry_mode)

        if 0 < self.config.incoming_edges < matrix.shape[0]:
            keep = np.zeros_like(matrix, dtype=bool)
            indices = np.argpartition(np.abs(matrix), -self.config.incoming_edges, axis=0)[-self.config.incoming_edges :]
            keep[indices, np.arange(matrix.shape[1])] = True
            matrix = np.where(keep, matrix, 0.0)

        if 0 < self.config.svd_rank < matrix.shape[0]:
            left, singular, right = np.linalg.svd(matrix, full_matrices=False)
            rank = self.config.svd_rank
            matrix = (left[:, :rank] * singular[:rank]) @ right[:rank]

        if self.config.column_normalization == "l1":
            matrix = matrix / np.maximum(np.sum(np.abs(matrix), axis=0, keepdims=True), 1e-12)
        elif self.config.column_normalization == "l2":
            matrix = matrix / np.maximum(np.sqrt(np.sum(matrix * matrix, axis=0, keepdims=True)), 1e-12)
        elif self.config.column_normalization != "none":
            raise ValueError(self.config.column_normalization)

        return matrix

    def getMyPosition(self, prcSoFar):
        prices = np.asarray(prcSoFar, dtype=float)
        n_instruments, n_days = prices.shape

        if self.last_day_count == n_days and self.last_position is not None:
            return self.last_position.copy()
        if self.last_day_count is not None and n_days < self.last_day_count:
            self.resetState()

        positions = np.zeros(n_instruments, dtype=int)
        returns = np.diff(np.log(prices), axis=1)
        minimum = max(60, max(lag for lag, _ in self.config.lag_weights) + 1)
        if n_instruments != 51 or returns.shape[1] < minimum:
            self.last_day_count = n_days
            self.last_position = positions
            return positions.copy()

        residual = returns[1:] - returns[0]
        if self.config.cross_sectional_demean:
            residual = residual - residual.mean(axis=0, keepdims=True)

        if self.config.freeze_at > 0 and residual.shape[1] > self.config.freeze_at:
            estimation = residual[:, : self.config.freeze_at]
        else:
            estimation = residual

        mean = estimation.mean(axis=1, keepdims=True)
        scale = np.maximum(estimation.std(axis=1, keepdims=True), 1e-12)
        z_estimation = (estimation - mean) / scale
        z_history = (residual - mean) / scale

        lead_signal = np.zeros(50)
        total_weight = 0.0
        for lag, weight in self.config.lag_weights:
            matrix = self._lead_matrix(z_estimation, lag)
            lead_signal += weight * (matrix.T @ z_history[:, -lag])
            total_weight += abs(weight)
        if total_weight > 0:
            lead_signal /= total_weight
        lead_signal = standardize_cross_section(lead_signal)

        reversal = -z_history[:, -self.config.reversal_lookback :].sum(axis=1)
        reversal = standardize_cross_section(reversal)
        if self.config.long_reversal_lookback > 0 and self.config.long_reversal_weight > 0:
            long_reversal = -z_history[:, -self.config.long_reversal_lookback :].sum(axis=1)
            long_reversal = standardize_cross_section(long_reversal)
            long_weight = self.config.long_reversal_weight
            reversal = standardize_cross_section((1.0 - long_weight) * reversal + long_weight * long_reversal)
        if self.config.reversal_mode == "linear":
            signal = (1.0 - self.config.reversal_weight) * lead_signal + self.config.reversal_weight * reversal
        elif self.config.reversal_mode == "agree_only":
            agreement = np.sign(lead_signal) == np.sign(reversal)
            blended = (1.0 - self.config.reversal_weight) * lead_signal + self.config.reversal_weight * reversal
            signal = np.where(agreement, blended, lead_signal)
        elif self.config.reversal_mode == "orthogonal":
            denominator = max(float(lead_signal @ lead_signal), 1e-12)
            independent_reversal = reversal - lead_signal * float(reversal @ lead_signal) / denominator
            independent_reversal = standardize_cross_section(independent_reversal)
            signal = (1.0 - self.config.reversal_weight) * lead_signal + self.config.reversal_weight * independent_reversal
        else:
            raise ValueError(self.config.reversal_mode)
        signal -= signal.mean()

        direction = np.sign(signal)
        if self.previous_direction is not None and self.config.hysteresis > 0:
            threshold = self.config.hysteresis * np.mean(np.abs(signal))
            if self.config.volatility_hysteresis:
                lookback = self.config.volatility_hysteresis_lookback
                asset_volatility = np.std(returns[1:, -lookback:], axis=1)
                relative_volatility = asset_volatility / max(float(np.median(asset_volatility)), 1e-12)
                multiplier = np.clip(relative_volatility, 0.5, 2.0) ** self.config.volatility_hysteresis_power
                threshold = threshold * multiplier
            weak = np.abs(signal) < threshold
            direction = np.where(weak, self.previous_direction, direction)

        if self.config.top_k < 50:
            keep = np.argsort(np.abs(signal))[-self.config.top_k :]
            active = np.zeros(50, dtype=bool)
            active[keep] = True
            direction = np.where(active, direction, 0.0)
        self.previous_direction = direction.copy()

        current = prices[:, -1]
        synthetic_limits = (ASSET_CAP / current[1:]).astype(int)
        positions[1:] = np.clip(direction.astype(int) * synthetic_limits, -synthetic_limits, synthetic_limits)
        hedge_betas = np.ones(50)
        if self.config.hedge_beta_shrinkage > 0:
            beta_fit = returns if self.config.hedge_beta_window <= 0 else returns[:, -self.config.hedge_beta_window :]
            factor = beta_fit[0] - beta_fit[0].mean()
            raw_betas = ((beta_fit[1:] - beta_fit[1:].mean(axis=1, keepdims=True)) @ factor) / max(float(factor @ factor), 1e-12)
            shrinkage = self.config.hedge_beta_shrinkage
            hedge_betas = 1.0 + shrinkage * (raw_betas - 1.0)
        factor_dollars = float((positions[1:] * current[1:]) @ hedge_betas)
        hedge_dollars = np.clip(-self.config.hedge_fraction * factor_dollars, -0.999 * ALGO_CAP, 0.999 * ALGO_CAP)
        algo_limit = int(ALGO_CAP / current[0])
        positions[0] = int(np.clip(np.trunc(hedge_dollars / current[0]), -algo_limit, algo_limit))

        self.last_day_count = n_days
        self.last_position = positions.astype(int)
        return positions.astype(int).copy()


def result_row(prices, label, config):
    strategy = ResearchLeadLag(config)
    official = backtest(prices, strategy, 249, 499)
    early = backtest(prices, strategy, 100, 249)
    folds = [backtest(prices, strategy, start, min(start + 50, 499))["score"] for start in range(100, 499, 50)]
    pseudo_windows = [
        backtest(prices, strategy, start, min(start + 250, 499))["score"]
        for start in (60, 100, 150, 200, 249)
    ]
    high_cost = backtest(prices, strategy, 249, 499, commission_multiplier=10.0)
    frozen = backtest(prices, ResearchLeadLag(replace(config, freeze_at=249)), 249, 499)
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
        "window_median": float(np.median(pseudo_windows)),
        "window_min": float(np.min(pseudo_windows)),
        "frozen_score": frozen["score"],
        "cost_10x_score": high_cost["score"],
        "volume": official["volume"],
        "config": repr(asdict(config)),
    }


def variants():
    baseline = Config()
    return [
        ("baseline", baseline),
        ("lag12_half", replace(baseline, lag_weights=((1, 1.0), (2, 0.5)))),
        ("lag12_equal", replace(baseline, lag_weights=((1, 1.0), (2, 1.0)))),
        ("lag123_decay", replace(baseline, lag_weights=((1, 1.0), (2, 0.5), (3, 0.25)))),
        ("lag123_equal", replace(baseline, lag_weights=((1, 1.0), (2, 1.0), (3, 1.0)))),
        ("lag123_slow", replace(baseline, lag_weights=((1, 1.0), (2, 0.75), (3, 0.5)))),
        ("lag1_to_5_decay", replace(baseline, lag_weights=((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.125), (5, 0.0625)))),
        ("cross_sectional_demean", replace(baseline, cross_sectional_demean=True)),
        ("soft_threshold", replace(baseline, soft_threshold=True)),
        ("recent250_25pct", replace(baseline, recent_window=250, recent_weight=0.25)),
        ("recent250_50pct", replace(baseline, recent_window=250, recent_weight=0.50)),
        ("recent120_25pct", replace(baseline, recent_window=120, recent_weight=0.25)),
        ("consensus250", replace(baseline, consensus_window=250)),
        ("consensus120", replace(baseline, consensus_window=120)),
        ("split_consensus", replace(baseline, split_consensus=True)),
        ("reversal_lb5", replace(baseline, reversal_lookback=5)),
        ("reversal_lb10", replace(baseline, reversal_lookback=10)),
        ("reversal_lb40", replace(baseline, reversal_lookback=40)),
        ("reversal_lb60", replace(baseline, reversal_lookback=60)),
        ("reversal_agree_only", replace(baseline, reversal_mode="agree_only")),
        ("reversal_orthogonal", replace(baseline, reversal_mode="orthogonal")),
        ("top45", replace(baseline, top_k=45)),
        ("top40", replace(baseline, top_k=40)),
        ("hysteresis_025", replace(baseline, hysteresis=0.25)),
        ("hysteresis_035", replace(baseline, hysteresis=0.35)),
        ("hedge_075", replace(baseline, hedge_fraction=0.75)),
        ("hedge_0875", replace(baseline, hedge_fraction=0.875)),
        ("incoming_3", replace(baseline, incoming_edges=3)),
        ("incoming_5", replace(baseline, incoming_edges=5)),
        ("incoming_10", replace(baseline, incoming_edges=10)),
        ("incoming_15", replace(baseline, incoming_edges=15)),
        ("incoming_20", replace(baseline, incoming_edges=20)),
        ("incoming_30", replace(baseline, incoming_edges=30)),
        ("asym_difference", replace(baseline, asymmetry_mode="difference")),
        ("asym_dominant", replace(baseline, asymmetry_mode="dominant")),
        ("asym_excess", replace(baseline, asymmetry_mode="excess")),
        ("svd_rank5", replace(baseline, svd_rank=5)),
        ("svd_rank10", replace(baseline, svd_rank=10)),
        ("svd_rank15", replace(baseline, svd_rank=15)),
        ("svd_rank20", replace(baseline, svd_rank=20)),
        ("svd_rank30", replace(baseline, svd_rank=30)),
        ("column_l1", replace(baseline, column_normalization="l1")),
        ("column_l2", replace(baseline, column_normalization="l2")),
        ("reversal_mix20_40_qtr", replace(baseline, long_reversal_lookback=40, long_reversal_weight=0.25)),
        ("reversal_mix20_40_half", replace(baseline, long_reversal_lookback=40, long_reversal_weight=0.50)),
        ("reversal_mix20_60_qtr", replace(baseline, long_reversal_lookback=60, long_reversal_weight=0.25)),
        ("reversal_mix20_60_half", replace(baseline, long_reversal_lookback=60, long_reversal_weight=0.50)),
    ]


def main():
    prices = pd.read_csv(ROOT / "prices.txt", sep=r"\s+").values.T
    rows = [result_row(prices, label, config) for label, config in variants()]
    results = pd.DataFrame(rows)
    results.to_csv(TABLES / "leadlag_improvement_lab.csv", index=False)
    columns = [
        "strategy",
        "official_score",
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
