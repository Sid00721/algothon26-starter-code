"""Residual lead-lag challenger for Negative Alpha, 14 July 2026.

The validated one-day cross-asset lead-lag network remains the alpha core.
This restrained challenger adds a small slow-reversal component and estimates
the ALGO hedge from causal 120-day single-index betas.
"""

import numpy as np


MIN_HISTORY = 60
SIGNIFICANCE_SE = 1.0
REVERSAL_WEIGHT = 0.20
REVERSAL_LOOKBACK = 20
LONG_REVERSAL_LOOKBACK = 60
LONG_REVERSAL_MIX = 0.075
HYSTERESIS_FRACTION = 0.30
HEDGE_BETA_LOOKBACK = 120

ASSET_CAP = 10_000.0
ALGO_CAP = 100_000.0
ALGO_HEDGE_CAP_FRACTION = 0.999

_previous_direction = None
_last_day_count = None
_last_position = None


def resetState():
    """Clear state between independent evaluation runs."""
    global _previous_direction, _last_day_count, _last_position
    _previous_direction = None
    _last_day_count = None
    _last_position = None


def _standardize_cross_section(values):
    scale = values.std()
    if scale < 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / scale


def getMyPosition(prcSoFar):
    """Return one legal integer position for each of the 51 instruments."""
    global _previous_direction, _last_day_count, _last_position

    prices = np.asarray(prcSoFar, dtype=float)
    n_instruments, n_days = prices.shape

    if _last_day_count == n_days and _last_position is not None:
        return _last_position.copy()
    if _last_day_count is not None and n_days < _last_day_count:
        resetState()

    positions = np.zeros(n_instruments, dtype=int)
    log_returns = np.diff(np.log(prices), axis=1)
    if n_instruments != 51 or log_returns.shape[1] < MIN_HISTORY:
        _last_day_count = n_days
        _last_position = positions
        return positions.copy()

    # Use ALGO-neutral returns to learn the diffuse one-day lead-lag network.
    residuals = log_returns[1:] - log_returns[0]
    mean = residuals.mean(axis=1, keepdims=True)
    scale = np.maximum(residuals.std(axis=1, keepdims=True), 1e-12)
    standardized = (residuals - mean) / scale

    predictors = standardized[:, :-1]
    targets = standardized[:, 1:]
    n_pairs = predictors.shape[1]
    lead_lag = predictors @ targets.T / n_pairs
    np.fill_diagonal(lead_lag, 0.0)
    lead_lag[np.abs(lead_lag) < SIGNIFICANCE_SE / np.sqrt(n_pairs)] = 0.0
    lead_signal = _standardize_cross_section(lead_lag.T @ standardized[:, -1])

    fast_reversal = _standardize_cross_section(
        -standardized[:, -REVERSAL_LOOKBACK:].sum(axis=1)
    )
    slow_reversal = _standardize_cross_section(
        -standardized[:, -LONG_REVERSAL_LOOKBACK:].sum(axis=1)
    )
    reversal = _standardize_cross_section(
        (1.0 - LONG_REVERSAL_MIX) * fast_reversal
        + LONG_REVERSAL_MIX * slow_reversal
    )
    signal = (1.0 - REVERSAL_WEIGHT) * lead_signal + REVERSAL_WEIGHT * reversal
    signal -= signal.mean()

    direction = np.sign(signal)
    if _previous_direction is not None:
        weak = np.abs(signal) < HYSTERESIS_FRACTION * np.mean(np.abs(signal))
        direction = np.where(weak, _previous_direction, direction)
    _previous_direction = direction.copy()

    current_prices = prices[:, -1]
    synthetic_limits = (ASSET_CAP / current_prices[1:]).astype(int)
    positions[1:] = direction.astype(int) * synthetic_limits
    positions[1:] = np.clip(positions[1:], -synthetic_limits, synthetic_limits)

    # Estimate heterogeneous ALGO loadings only for the hedge.  The alpha
    # model remains the stable beta-one residual specification.
    beta_sample = log_returns[:, -HEDGE_BETA_LOOKBACK:]
    factor = beta_sample[0] - beta_sample[0].mean()
    denominator = max(float(factor @ factor), 1e-12)
    hedge_betas = (
        (beta_sample[1:] - beta_sample[1:].mean(axis=1, keepdims=True)) @ factor
    ) / denominator
    factor_dollars = float((positions[1:] * current_prices[1:]) @ hedge_betas)
    hedge_cap = ALGO_HEDGE_CAP_FRACTION * ALGO_CAP
    hedge_dollars = np.clip(-factor_dollars, -hedge_cap, hedge_cap)
    algo_limit = int(ALGO_CAP / current_prices[0])
    positions[0] = int(np.trunc(hedge_dollars / current_prices[0]))
    positions[0] = int(np.clip(positions[0], -algo_limit, algo_limit))

    _last_day_count = n_days
    _last_position = positions.astype(int)
    return positions.astype(int).copy()
