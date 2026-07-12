"""Clean-room aggregated lead-lag candidate for Algothon 2026 research.

This module is deliberately separate from ``teamName.py``.  It predicts the
next ALGO-neutral residual return of each synthetic asset from the latest
residual returns of all the other synthetic assets.  ALGO is used only to
hedge the resulting net market exposure.
"""

import numpy as np


MIN_HISTORY = 60
SIGNIFICANCE_SE = 1.0
REVERSAL_WEIGHT = 0.20
REVERSAL_LOOKBACK = 20
HYSTERESIS_FRACTION = 0.30
SIZING_MODE = "sign"  # alternatives used by the harness: "tanh", "rank"
ESTIMATION_WINDOW = 0  # zero means use all information available so far
FREEZE_ESTIMATION_AT = 0  # research-only: freeze the matrix after N returns
HEDGE_FRACTION = 1.0

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
    if log_returns.shape[1] < MIN_HISTORY or n_instruments != 51:
        _last_day_count = n_days
        _last_position = positions
        return positions.copy()

    # Remove the contemporaneous common factor without estimating 50 unstable
    # beta coefficients.  Only assets 1..50 participate in alpha generation.
    residual_history = log_returns[1:] - log_returns[0]
    if FREEZE_ESTIMATION_AT > 0 and residual_history.shape[1] > FREEZE_ESTIMATION_AT:
        estimation_sample = residual_history[:, :FREEZE_ESTIMATION_AT]
    elif ESTIMATION_WINDOW > 0:
        estimation_sample = residual_history[:, -ESTIMATION_WINDOW:]
    else:
        estimation_sample = residual_history

    time_mean = estimation_sample.mean(axis=1, keepdims=True)
    time_scale = estimation_sample.std(axis=1, keepdims=True)
    time_scale = np.maximum(time_scale, 1e-12)
    standardized_estimation = (estimation_sample - time_mean) / time_scale
    standardized_history = (residual_history - time_mean) / time_scale

    predictors = standardized_estimation[:, :-1]
    targets = standardized_estimation[:, 1:]
    n_pairs = predictors.shape[1]
    lead_lag = predictors @ targets.T / n_pairs
    np.fill_diagonal(lead_lag, 0.0)

    noise_floor = SIGNIFICANCE_SE / np.sqrt(n_pairs)
    lead_lag[np.abs(lead_lag) < noise_floor] = 0.0
    lead_signal = lead_lag.T @ standardized_history[:, -1]
    lead_signal = _standardize_cross_section(lead_signal)

    reversal_signal = -standardized_history[:, -REVERSAL_LOOKBACK:].sum(axis=1)
    reversal_signal = _standardize_cross_section(reversal_signal)
    signal = (1.0 - REVERSAL_WEIGHT) * lead_signal + REVERSAL_WEIGHT * reversal_signal
    signal -= signal.mean()

    if SIZING_MODE == "sign":
        scaled_target = np.sign(signal)
    elif SIZING_MODE == "tanh":
        scaled_target = np.tanh(signal)
    elif SIZING_MODE == "rank":
        ranks = np.argsort(np.argsort(signal))
        scaled_target = (ranks - 0.5 * (len(signal) - 1)) / (0.5 * (len(signal) - 1))
    else:
        raise ValueError(f"unsupported SIZING_MODE: {SIZING_MODE}")

    # Weak updates keep their previous direction.  Strong updates are always
    # allowed to flip, so this is hysteresis rather than a fixed rebalance rule.
    current_direction = np.sign(scaled_target)
    if _previous_direction is not None and HYSTERESIS_FRACTION > 0:
        threshold = HYSTERESIS_FRACTION * np.mean(np.abs(signal))
        weak_update = np.abs(signal) < threshold
        current_direction = np.where(weak_update, _previous_direction, current_direction)
        scaled_target = np.where(weak_update, current_direction, scaled_target)
    _previous_direction = current_direction.copy()

    current_prices = prices[:, -1]
    synthetic_limits = (ASSET_CAP / current_prices[1:]).astype(int)
    positions[1:] = np.rint(scaled_target * synthetic_limits).astype(int)
    positions[1:] = np.clip(positions[1:], -synthetic_limits, synthetic_limits)

    # Hedge the realised post-rounding dollar exposure, not the pre-rounding
    # signal, and respect ALGO's moving daily share limit.
    synthetic_dollars = positions[1:] * current_prices[1:]
    hedge_dollars = -HEDGE_FRACTION * synthetic_dollars.sum()
    hedge_cap = ALGO_HEDGE_CAP_FRACTION * ALGO_CAP
    hedge_dollars = np.clip(hedge_dollars, -hedge_cap, hedge_cap)
    algo_limit = int(ALGO_CAP / current_prices[0])
    positions[0] = int(np.trunc(hedge_dollars / current_prices[0]))
    positions[0] = int(np.clip(positions[0], -algo_limit, algo_limit))

    _last_day_count = n_days
    _last_position = positions.astype(int)
    return positions.astype(int).copy()
