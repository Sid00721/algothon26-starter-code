"""Algothon 2026 submission strategy.

The portfolio combines volatility-normalised 5/20-day reversal with two
stable relative-value relationships.  All sleeve signals are netted before
one integer position is produced for each instrument.
"""

import numpy as np


VOLATILITY_LOOKBACK = 60
SHORT_HORIZON = 5
MEDIUM_HORIZON = 20
SHORT_WEIGHT = 0.50
MEDIUM_WEIGHT = 0.50
SIGNAL_STRENGTH = 4.0
SIGNAL_THRESHOLD = 0.25

PAIR_LOOKBACK = 120
PAIR_Z_LOOKBACK = 60
PAIR_WEIGHT = 3.0
PAIRS = ((1, 20), (8, 27))  # AENO/NWIG and HUXZ/ACAC

DEADBAND_FRACTION = 0.01

_last_day_count = None
_last_position = None


def resetState():
    """Reset the small no-trade-band cache (useful for independent backtests)."""
    global _last_day_count, _last_position
    _last_day_count = None
    _last_position = None


def getMyPosition(prcSoFar):
    """Return one capped integer target position for every instrument."""
    global _last_day_count, _last_position

    prices = np.asarray(prcSoFar, dtype=float)
    n_instruments, n_days = prices.shape

    # Repeated evaluation of the same day must not advance internal state.
    if _last_day_count == n_days and _last_position is not None:
        return _last_position.copy()
    if _last_day_count is not None and n_days < _last_day_count:
        resetState()

    if n_days <= MEDIUM_HORIZON:
        target_shares = np.zeros(n_instruments, dtype=int)
        _last_day_count = n_days
        _last_position = target_shares
        return target_shares.copy()

    history_start = max(0, n_days - VOLATILITY_LOOKBACK - 1)
    log_returns = np.diff(np.log(prices[:, history_start:]), axis=1)
    volatility = np.maximum(np.std(log_returns, axis=1, ddof=1), 1e-6)

    short_reversal = -np.log(prices[:, -1] / prices[:, -1 - SHORT_HORIZON])
    short_reversal /= volatility * np.sqrt(SHORT_HORIZON)
    medium_reversal = -np.log(prices[:, -1] / prices[:, -1 - MEDIUM_HORIZON])
    medium_reversal /= volatility * np.sqrt(MEDIUM_HORIZON)
    signal = SHORT_WEIGHT * short_reversal + MEDIUM_WEIGHT * medium_reversal

    # Add causal spread z-scores.  The correct ACAC column is index 27.
    if n_days >= PAIR_LOOKBACK:
        log_prices = np.log(prices[:, -PAIR_LOOKBACK:])
        for first, second in PAIRS:
            x = log_prices[second]
            y = log_prices[first]
            x_centered = x - x.mean()
            denominator = max(np.dot(x_centered, x_centered), 1e-12)
            beta = np.dot(x_centered, y - y.mean()) / denominator
            spread = y - beta * x
            recent_spread = spread[-PAIR_Z_LOOKBACK:]
            spread_std = recent_spread.std(ddof=1)
            if spread_std > 1e-8:
                z_score = (spread[-1] - recent_spread.mean()) / spread_std
                signal[first] -= PAIR_WEIGHT * z_score
                signal[second] += PAIR_WEIGHT * np.sign(beta) * z_score

    # Weak signals are more likely to be noise than alpha.
    active_signal = np.where(np.abs(signal) >= SIGNAL_THRESHOLD, signal, 0.0)

    dollar_limits = np.full(n_instruments, 10_000.0)
    if n_instruments:
        dollar_limits[0] = 100_000.0
    target_dollars = dollar_limits * np.tanh(SIGNAL_STRENGTH * active_signal)

    current_prices = prices[:, -1]
    share_limits = (dollar_limits / current_prices).astype(int)
    target_shares = np.rint(target_dollars / current_prices).astype(int)
    target_shares = np.clip(target_shares, -share_limits, share_limits).astype(int)

    # Retain the legal prior position when the requested trade is under 1% of
    # that instrument's dollar cap.  This suppresses small commission churn.
    if _last_day_count == n_days - 1 and _last_position is not None:
        held = np.clip(_last_position, -share_limits, share_limits).astype(int)
        trade_dollars = np.abs(target_shares - held) * current_prices
        small_trade = trade_dollars <= DEADBAND_FRACTION * dollar_limits
        target_shares = np.where(small_trade, held, target_shares).astype(int)

    _last_day_count = n_days
    _last_position = target_shares.copy()
    return target_shares.astype(int)
