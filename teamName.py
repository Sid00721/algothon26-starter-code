"""Algothon 2026 submission strategy.

The model is a volatility-normalised, multi-horizon cross-sectional reversal
strategy.  It is deliberately stateless so repeated calls with the same price
history always produce the same desired portfolio.
"""

import numpy as np


N_INSTRUMENTS = 51
VOLATILITY_LOOKBACK = 60
SHORT_HORIZON = 5
MEDIUM_HORIZON = 20
SHORT_WEIGHT = 0.20
MEDIUM_WEIGHT = 0.80
SIGNAL_STRENGTH = 3.0


def getMyPosition(prcSoFar):
    """Return the desired integer share position for every instrument.

    Only information available through the latest column is used.  Recent
    returns are scaled by their trailing volatility, which makes signals from
    assets with different price and risk levels comparable.  The smooth tanh
    mapping avoids a brittle all-or-nothing threshold while respecting the
    competition's per-instrument dollar limits.
    """
    prices = np.asarray(prcSoFar, dtype=float)
    n_instruments, n_days = prices.shape

    if n_days <= MEDIUM_HORIZON:
        return np.zeros(n_instruments, dtype=int)

    history_start = max(0, n_days - VOLATILITY_LOOKBACK - 1)
    recent_prices = prices[:, history_start:]
    log_returns = np.diff(np.log(recent_prices), axis=1)
    volatility = np.std(log_returns, axis=1, ddof=1)
    volatility = np.maximum(volatility, 1e-6)

    short_reversal = -np.log(prices[:, -1] / prices[:, -1 - SHORT_HORIZON])
    short_reversal /= volatility * np.sqrt(SHORT_HORIZON)

    medium_reversal = -np.log(prices[:, -1] / prices[:, -1 - MEDIUM_HORIZON])
    medium_reversal /= volatility * np.sqrt(MEDIUM_HORIZON)

    signal = SHORT_WEIGHT * short_reversal + MEDIUM_WEIGHT * medium_reversal

    dollar_limits = np.full(n_instruments, 10_000.0)
    if n_instruments:
        dollar_limits[0] = 100_000.0

    target_dollars = dollar_limits * np.tanh(SIGNAL_STRENGTH * signal)
    target_shares = np.rint(target_dollars / prices[:, -1]).astype(int)
    share_limits = (dollar_limits / prices[:, -1]).astype(int)
    target_shares = np.clip(target_shares, -share_limits, share_limits)

    return target_shares
