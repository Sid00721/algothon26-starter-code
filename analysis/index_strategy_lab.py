#!/usr/bin/env python3
"""Causal index/stat-arb strategy laboratory for Algothon 2026.

The models here are research candidates, not submission files.  They test the
specific hypotheses raised by the index-arbitrage prompt and the attached
Avellaneda-Lee paper against the saved aggregated lead-lag benchmark.
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

from analysis.stress_final_strategy import backtest


ASSET_CAP = 10_000.0
ALGO_CAP = 100_000.0


def _standardize_rows(values):
    mean = values.mean(axis=1, keepdims=True)
    scale = np.maximum(values.std(axis=1, keepdims=True), 1e-12)
    return (values - mean) / scale


def _standardize_cross_section(values):
    scale = values.std()
    return np.zeros_like(values) if scale < 1e-12 else (values - values.mean()) / scale


def _signal_positions(signal, prices, previous_direction, band=0.30, sizing="sign", hedge=True):
    signal = signal - signal.mean()
    if sizing == "sign":
        scaled = np.sign(signal)
    elif sizing == "tanh":
        scaled = np.tanh(signal)
    elif sizing == "rank":
        ranks = np.argsort(np.argsort(signal))
        scaled = (ranks - 0.5 * (len(signal) - 1)) / (0.5 * (len(signal) - 1))
    else:
        raise ValueError(sizing)

    direction = np.sign(scaled)
    if previous_direction is not None and band > 0:
        weak = np.abs(signal) < band * np.mean(np.abs(signal))
        direction = np.where(weak, previous_direction, direction)
        scaled = np.where(weak, direction, scaled)

    output = np.zeros(51, dtype=int)
    limits = (ASSET_CAP / prices[1:]).astype(int)
    output[1:] = np.clip(np.rint(scaled * limits).astype(int), -limits, limits)
    if hedge:
        net_synthetic = float(output[1:] @ prices[1:])
        hedge_dollars = np.clip(-net_synthetic, -0.999 * ALGO_CAP, 0.999 * ALGO_CAP)
        algo_limit = int(ALGO_CAP / prices[0])
        output[0] = int(np.clip(np.trunc(hedge_dollars / prices[0]), -algo_limit, algo_limit))
    return output, direction


class StatefulStrategy:
    def resetState(self):
        self.previous_direction = None
        self.spread_direction = 0


class IndexBasketOU(StatefulStrategy):
    """Trade an ALGO/geometric-basket residual only when its OU fit is valid."""

    def __init__(self, beta_window=120, ou_window=60, entry=1.25, exit_level=0.50):
        self.beta_window = beta_window
        self.ou_window = ou_window
        self.entry = entry
        self.exit_level = exit_level
        self.resetState()

    def getMyPosition(self, prices):
        prices = np.asarray(prices, dtype=float)
        returns = np.diff(np.log(prices), axis=1)
        output = np.zeros(51, dtype=int)
        if returns.shape[1] < max(self.beta_window, self.ou_window):
            return output

        sample = returns[:, -self.beta_window :]
        basket = sample[1:].mean(axis=0)
        index = sample[0]
        basket_centered = basket - basket.mean()
        beta = np.dot(basket_centered, index - index.mean()) / max(np.dot(basket_centered, basket_centered), 1e-12)
        intercept = index.mean() - beta * basket.mean()
        residual_returns = index - intercept - beta * basket
        residual_path = np.cumsum(residual_returns[-self.ou_window :])

        x = residual_path[:-1]
        y = residual_path[1:]
        x_centered = x - x.mean()
        ar = np.dot(x_centered, y - y.mean()) / max(np.dot(x_centered, x_centered), 1e-12)
        ar_intercept = y.mean() - ar * x.mean()
        innovations = y - ar_intercept - ar * x
        valid = 0.0 < ar < 0.9672
        if valid:
            equilibrium = ar_intercept / (1.0 - ar)
            equilibrium_scale = np.sqrt(np.var(innovations) / max(1.0 - ar * ar, 1e-12))
            s_score = (residual_path[-1] - equilibrium) / max(equilibrium_scale, 1e-12)
            if self.spread_direction == 0:
                if s_score > self.entry:
                    self.spread_direction = -1
                elif s_score < -self.entry:
                    self.spread_direction = 1
            elif self.spread_direction > 0 and s_score > -self.exit_level:
                self.spread_direction = 0
            elif self.spread_direction < 0 and s_score < self.exit_level:
                self.spread_direction = 0
        else:
            self.spread_direction = 0

        current = prices[:, -1]
        algo_limit = int(ALGO_CAP / current[0])
        output[0] = self.spread_direction * algo_limit
        basket_dollars = -self.spread_direction * min(abs(beta), 1.0) * ALGO_CAP / 50.0
        output[1:] = np.rint(basket_dollars / current[1:]).astype(int)
        synthetic_limits = (ASSET_CAP / current[1:]).astype(int)
        output[1:] = np.clip(output[1:], -synthetic_limits, synthetic_limits)
        return output.astype(int)


class PCAResidualOU(StatefulStrategy):
    """Avellaneda-Lee-style PCA residual OU strategy on assets 1..50."""

    def __init__(self, window=90, ou_window=60, factors=3, entry=1.25, exit_level=0.50):
        self.window = window
        self.ou_window = ou_window
        self.factors = factors
        self.entry = entry
        self.exit_level = exit_level
        self.resetState()

    def resetState(self):
        super().resetState()
        self.asset_direction = np.zeros(50)

    def getMyPosition(self, prices):
        prices = np.asarray(prices, dtype=float)
        returns = np.diff(np.log(prices[1:]), axis=1)
        if returns.shape[1] < self.window:
            return np.zeros(51, dtype=int)
        sample = returns[:, -self.window :]
        mean = sample.mean(axis=1, keepdims=True)
        scale = np.maximum(sample.std(axis=1, keepdims=True), 1e-12)
        standardized = (sample - mean) / scale
        covariance = standardized @ standardized.T / standardized.shape[1]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        loadings = eigenvectors[:, np.argsort(eigenvalues)[::-1][: self.factors]]
        residual_returns = (standardized - loadings @ (loadings.T @ standardized)) * scale
        residual_path = np.cumsum(residual_returns[:, -self.ou_window :], axis=1)

        x = residual_path[:, :-1]
        y = residual_path[:, 1:]
        x_mean = x.mean(axis=1)
        y_mean = y.mean(axis=1)
        x_centered = x - x_mean[:, None]
        denominator = np.sum(x_centered * x_centered, axis=1)
        ar = np.sum(x_centered * (y - y_mean[:, None]), axis=1) / np.maximum(denominator, 1e-12)
        intercept = y_mean - ar * x_mean
        innovations = y - intercept[:, None] - ar[:, None] * x
        equilibrium = intercept / np.maximum(1.0 - ar, 1e-12)
        equilibrium_scale = np.sqrt(np.var(innovations, axis=1) / np.maximum(1.0 - ar * ar, 1e-12))
        s_score = (residual_path[:, -1] - equilibrium) / np.maximum(equilibrium_scale, 1e-12)
        valid = (ar > 0.0) & (ar < 0.9672) & np.isfinite(s_score)

        open_long = valid & (s_score < -self.entry)
        open_short = valid & (s_score > self.entry)
        close_long = (self.asset_direction > 0) & (s_score > -self.exit_level)
        close_short = (self.asset_direction < 0) & (s_score < self.exit_level)
        self.asset_direction[close_long | close_short | ~valid] = 0
        self.asset_direction[open_long] = 1
        self.asset_direction[open_short] = -1
        output, _ = _signal_positions(self.asset_direction, prices[:, -1], None, band=0, sizing="sign", hedge=True)
        inactive = self.asset_direction == 0
        output[1:][inactive] = 0
        # Recalculate ALGO hedge after inactive positions are removed.
        net = float(output[1:] @ prices[1:, -1])
        output[0] = int(np.trunc(np.clip(-net, -0.999 * ALGO_CAP, 0.999 * ALGO_CAP) / prices[0, -1]))
        return output.astype(int)


class IndexAugmentedLeadLag(StatefulStrategy):
    """Residual lead-lag with optional ALGO->residual and residual->ALGO paths."""

    def __init__(self, index_predictor=True, factor_weight=0.0, reversal_weight=0.20, threshold_se=1.0, band=0.30):
        self.index_predictor = index_predictor
        self.factor_weight = factor_weight
        self.reversal_weight = reversal_weight
        self.threshold_se = threshold_se
        self.band = band
        self.resetState()

    def getMyPosition(self, prices):
        prices = np.asarray(prices, dtype=float)
        returns = np.diff(np.log(prices), axis=1)
        if returns.shape[1] < 60:
            return np.zeros(51, dtype=int)
        residuals = returns[1:] - returns[0]
        z = _standardize_rows(residuals)
        index_z = (returns[0] - returns[0].mean()) / max(returns[0].std(), 1e-12)
        predictors = np.vstack([z, index_z]) if self.index_predictor else z
        n_pairs = predictors.shape[1] - 1
        matrix = predictors[:, :-1] @ z[:, 1:].T / n_pairs
        np.fill_diagonal(matrix[:50], 0.0)
        matrix[np.abs(matrix) < self.threshold_se / np.sqrt(n_pairs)] = 0.0
        lead_signal = _standardize_cross_section(matrix.T @ predictors[:, -1])
        reversal = _standardize_cross_section(-z[:, -20:].sum(axis=1))
        signal = (1.0 - self.reversal_weight) * lead_signal + self.reversal_weight * reversal
        output, self.previous_direction = _signal_positions(signal, prices[:, -1], self.previous_direction, self.band, "sign", True)

        if self.factor_weight > 0:
            factor_links = z[:, :-1] @ index_z[1:] / n_pairs
            factor_links[np.abs(factor_links) < self.threshold_se / np.sqrt(n_pairs)] = 0.0
            factor_signal = float(factor_links @ z[:, -1])
            alpha_dollars = self.factor_weight * ALGO_CAP * np.tanh(factor_signal)
            algo_limit = int(ALGO_CAP / prices[0, -1])
            output[0] = int(np.clip(output[0] + np.trunc(alpha_dollars / prices[0, -1]), -algo_limit, algo_limit))
        return output.astype(int)


class BetaResidualLeadLag(StatefulStrategy):
    """Lead-lag after causal single-index beta residualization.

    This is the direct factor-neutral interpretation of the Avellaneda-Lee
    framework.  ``beta_shrinkage=0`` reduces to the candidate's beta-one
    residual; larger values estimate heterogeneous ALGO loadings.
    """

    def __init__(
        self,
        beta_window=0,
        beta_shrinkage=1.0,
        reversal_weight=0.20,
        threshold_se=1.0,
        band=0.30,
    ):
        self.beta_window = beta_window
        self.beta_shrinkage = beta_shrinkage
        self.reversal_weight = reversal_weight
        self.threshold_se = threshold_se
        self.band = band
        self.resetState()

    def getMyPosition(self, prices):
        prices = np.asarray(prices, dtype=float)
        returns = np.diff(np.log(prices), axis=1)
        if returns.shape[1] < 60:
            return np.zeros(51, dtype=int)

        fit = returns if self.beta_window <= 0 else returns[:, -self.beta_window :]
        factor = fit[0] - fit[0].mean()
        raw_beta = ((fit[1:] - fit[1:].mean(axis=1, keepdims=True)) @ factor) / max(factor @ factor, 1e-12)
        beta = 1.0 + self.beta_shrinkage * (raw_beta - 1.0)

        residuals = returns[1:] - beta[:, None] * returns[0]
        z = _standardize_rows(residuals)
        n_pairs = z.shape[1] - 1
        matrix = z[:, :-1] @ z[:, 1:].T / n_pairs
        np.fill_diagonal(matrix, 0.0)
        matrix[np.abs(matrix) < self.threshold_se / np.sqrt(n_pairs)] = 0.0
        lead_signal = _standardize_cross_section(matrix.T @ z[:, -1])
        reversal = _standardize_cross_section(-z[:, -20:].sum(axis=1))
        signal = (1.0 - self.reversal_weight) * lead_signal + self.reversal_weight * reversal

        output, self.previous_direction = _signal_positions(
            signal,
            prices[:, -1],
            self.previous_direction,
            self.band,
            "sign",
            hedge=False,
        )
        factor_dollars = float((output[1:] * prices[1:, -1]) @ beta)
        algo_limit = int(ALGO_CAP / prices[0, -1])
        output[0] = int(
            np.clip(
                np.trunc(np.clip(-factor_dollars, -0.999 * ALGO_CAP, 0.999 * ALGO_CAP) / prices[0, -1]),
                -algo_limit,
                algo_limit,
            )
        )
        return output.astype(int)


class RidgeResidualVAR(StatefulStrategy):
    """Dense ridge VAR(1) alternative to thresholded lead-lag correlations."""

    def __init__(self, ridge=0.10, reversal_weight=0.20, band=0.30):
        self.ridge = ridge
        self.reversal_weight = reversal_weight
        self.band = band
        self.resetState()

    def getMyPosition(self, prices):
        prices = np.asarray(prices, dtype=float)
        returns = np.diff(np.log(prices), axis=1)
        if returns.shape[1] < 60:
            return np.zeros(51, dtype=int)
        z = _standardize_rows(returns[1:] - returns[0])
        x = z[:, :-1].T
        y = z[:, 1:].T
        gram = x.T @ x
        penalty = self.ridge * np.trace(gram) / gram.shape[0]
        coefficients = np.linalg.solve(gram + penalty * np.eye(gram.shape[0]), x.T @ y)
        lead_signal = _standardize_cross_section(z[:, -1] @ coefficients)
        reversal = _standardize_cross_section(-z[:, -20:].sum(axis=1))
        signal = (1.0 - self.reversal_weight) * lead_signal + self.reversal_weight * reversal
        output, self.previous_direction = _signal_positions(signal, prices[:, -1], self.previous_direction, self.band, "sign", True)
        return output.astype(int)


class PCAResidualLeadLag(StatefulStrategy):
    """Remove several PCA factors, then estimate aggregated residual lead-lag."""

    def __init__(self, factors=1, threshold_se=1.0, band=0.30):
        self.factors = factors
        self.threshold_se = threshold_se
        self.band = band
        self.resetState()

    def getMyPosition(self, prices):
        prices = np.asarray(prices, dtype=float)
        returns = np.diff(np.log(prices[1:]), axis=1)
        if returns.shape[1] < 60:
            return np.zeros(51, dtype=int)
        z = _standardize_rows(returns)
        covariance = z @ z.T / z.shape[1]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        loadings = eigenvectors[:, np.argsort(eigenvalues)[::-1][: self.factors]]
        residual = z - loadings @ (loadings.T @ z)
        n_pairs = residual.shape[1] - 1
        matrix = residual[:, :-1] @ residual[:, 1:].T / n_pairs
        np.fill_diagonal(matrix, 0.0)
        matrix[np.abs(matrix) < self.threshold_se / np.sqrt(n_pairs)] = 0.0
        signal = matrix.T @ residual[:, -1]
        output, self.previous_direction = _signal_positions(signal, prices[:, -1], self.previous_direction, self.band, "sign", True)
        return output.astype(int)


def evaluate(prices, label, strategy):
    early = backtest(prices, strategy, 100, 249)
    official = backtest(prices, strategy, 249, 499)
    folds = [backtest(prices, strategy, start, min(start + 50, 499))["score"] for start in range(100, 499, 50)]
    return {
        "strategy": label,
        "official_score": official["score"],
        "official_mean": official["mean_pnl"],
        "official_std": official["std_pnl"],
        "official_sharpe": official["sharpe"],
        "early_score": early["score"],
        "fold_median": float(np.median(folds)),
        "fold_min": float(np.min(folds)),
        "positive_folds": int(np.sum(np.asarray(folds) > 0)),
        "volume": official["volume"],
    }


def main():
    prices = pd.read_csv(ROOT / "prices.txt", sep=r"\s+").values.T
    benchmark = importlib.import_module("analysis.candidate_leadlag")
    strategies = [("saved_leadlag", benchmark)]
    strategies += [(f"index_basket_OU_b{beta}_o{ou}", IndexBasketOU(beta, ou)) for beta, ou in ((60, 60), (120, 60), (180, 60), (180, 90))]
    strategies += [(f"PCA_OU_k{k}", PCAResidualOU(factors=k)) for k in (1, 3, 5, 10, 15)]
    strategies += [
        ("leadlag_plus_index_predictor", IndexAugmentedLeadLag(index_predictor=True, factor_weight=0.0)),
        ("leadlag_no_index_predictor", IndexAugmentedLeadLag(index_predictor=False, factor_weight=0.0)),
        ("bidirectional_factor_0.25", IndexAugmentedLeadLag(index_predictor=True, factor_weight=0.25)),
        ("bidirectional_factor_0.50", IndexAugmentedLeadLag(index_predictor=True, factor_weight=0.50)),
    ]
    strategies += [
        (f"beta_residual_w{window}_s{shrinkage}", BetaResidualLeadLag(window, shrinkage))
        for window in (0, 120, 250)
        for shrinkage in (0.50, 1.00)
    ]
    strategies += [(f"ridge_VAR_{ridge}", RidgeResidualVAR(ridge=ridge)) for ridge in (0.01, 0.10, 1.0, 10.0)]
    strategies += [(f"PCA_leadlag_k{k}", PCAResidualLeadLag(factors=k)) for k in (1, 2, 3, 5, 10)]

    results = pd.DataFrame([evaluate(prices, label, strategy) for label, strategy in strategies])
    output = ROOT / "analysis" / "output" / "tables" / "index_strategy_lab.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print(results.sort_values("official_score", ascending=False).to_string(index=False, float_format=lambda value: f"{value:,.2f}"))


if __name__ == "__main__":
    main()
