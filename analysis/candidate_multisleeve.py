"""Competition-compatible multi-sleeve statistical-arbitrage candidate.

This file is intentionally kept separate from teamName.py until the candidate
has passed the stress tests in analysis/stress_multisleeve.py.
"""

import numpy as np


N_INSTRUMENTS = 51
BETA_LOOKBACK = 120
PAIR_Z_LOOKBACK = 60
REVERSAL_HORIZONS = (20, 80, 120)
REVERSAL_WEIGHTS = (0.25, 0.25, 0.50)
REVERSAL_SLEEVE_WEIGHT = 0.60
PAIR_SLEEVE_WEIGHT = 0.40
HEDGE_FRACTION = 1.0
SIGNAL_STRENGTH = 2.0
PAIR_ENTRY_Z = 1.25
PAIR_EXIT_Z = 0.25
DEADBAND_FRACTION = 0.01

# Zero-based indices from prices.txt. ACAC is 27; index 30 is RCRI.
PAIRS = ((1, 20), (8, 27))  # AENO/NWIG and HUXZ/ACAC

_current_positions = np.zeros(N_INSTRUMENTS, dtype=int)
_pair_states = np.zeros(len(PAIRS), dtype=int)
_last_num_days = -1
_last_output = np.zeros(N_INSTRUMENTS, dtype=int)


def resetState():
    """Reset state for a fresh evaluation run (used only by research tests)."""
    global _current_positions, _pair_states, _last_num_days, _last_output
    _current_positions = np.zeros(N_INSTRUMENTS, dtype=int)
    _pair_states = np.zeros(len(PAIRS), dtype=int)
    _last_num_days = -1
    _last_output = np.zeros(N_INSTRUMENTS, dtype=int)


def _cold_start_positions(prices, dollar_limits):
    """Conservative fallback used only when fewer than 120 days are present."""
    n_instruments, n_days = prices.shape
    if n_days < 6:
        return np.zeros(n_instruments, dtype=int)

    five_day_reversal = -np.log(prices[:, -1] / prices[:, -6])
    recent_returns = np.diff(np.log(prices[:, -min(n_days, 21) :]), axis=1)
    volatility = np.maximum(np.std(recent_returns, axis=1, ddof=1), 1e-6)
    signal = five_day_reversal / (volatility * np.sqrt(5.0))
    target_dollars = 0.20 * dollar_limits * np.tanh(signal)
    return np.rint(target_dollars / prices[:, -1]).astype(int)


def getMyPosition(prcSoFar):
    """Return a net vector of 51 integer target share positions."""
    global _current_positions, _pair_states, _last_num_days, _last_output

    prices = np.asarray(prcSoFar, dtype=float)
    n_instruments, n_days = prices.shape

    if n_instruments != N_INSTRUMENTS:
        return np.zeros(n_instruments, dtype=int)

    # Make repeated calls for the same day idempotent. Reset if a new evaluator
    # starts from an earlier history length in the same Python process.
    if n_days == _last_num_days:
        return _last_output.copy()
    if n_days < _last_num_days:
        resetState()

    current_prices = prices[:, -1]
    dollar_limits = np.full(n_instruments, 10_000.0)
    dollar_limits[0] = 100_000.0
    max_shares = (dollar_limits / current_prices).astype(int)

    if n_days <= BETA_LOOKBACK:
        target_positions = _cold_start_positions(prices, dollar_limits)
    else:
        log_prices = np.log(prices[:, -(BETA_LOOKBACK + 1) :])
        returns = np.diff(log_prices, axis=1)

        algo_returns = returns[0]
        algo_centered = algo_returns - np.mean(algo_returns)
        synthetic_centered = returns[1:] - np.mean(returns[1:], axis=1, keepdims=True)
        beta_denominator = float(algo_centered @ algo_centered) + 1e-12
        betas = (synthetic_centered @ algo_centered) / beta_denominator
        betas = np.clip(betas, -3.0, 3.0)

        residual_returns = returns[1:] - betas[:, None] * algo_returns[None, :]
        residual_volatility = np.maximum(np.std(residual_returns, axis=1, ddof=1), 1e-6)

        reversal_signal = np.zeros(n_instruments - 1)
        for horizon, weight in zip(REVERSAL_HORIZONS, REVERSAL_WEIGHTS):
            cumulative_residual = np.sum(residual_returns[:, -horizon:], axis=1)
            standardized_reversal = -cumulative_residual / (
                residual_volatility * np.sqrt(float(horizon))
            )
            reversal_signal += weight * standardized_reversal

        inverse_volatility = np.median(residual_volatility) / residual_volatility
        inverse_volatility = np.clip(inverse_volatility, 0.50, 1.50)
        reversal_dollars = (
            REVERSAL_SLEEVE_WEIGHT
            * 10_000.0
            * inverse_volatility
            * np.tanh(SIGNAL_STRENGTH * reversal_signal)
        )

        pair_dollars = np.zeros(n_instruments - 1)
        pair_log_prices = np.log(prices[:, -BETA_LOOKBACK:])
        for pair_number, (left_index, right_index) in enumerate(PAIRS):
            left = pair_log_prices[left_index]
            right = pair_log_prices[right_index]
            right_centered = right - np.mean(right)
            hedge_ratio = float((left - np.mean(left)) @ right_centered) / (
                float(right_centered @ right_centered) + 1e-12
            )
            intercept = float(np.mean(left) - hedge_ratio * np.mean(right))
            spread = left - intercept - hedge_ratio * right
            spread_window = spread[-PAIR_Z_LOOKBACK:]
            spread_std = float(np.std(spread_window, ddof=1))
            z_score = 0.0
            if spread_std > 1e-10:
                z_score = float((spread[-1] - np.mean(spread_window)) / spread_std)

            state = int(_pair_states[pair_number])
            if state == 0:
                if z_score > PAIR_ENTRY_Z:
                    state = -1  # short the spread
                elif z_score < -PAIR_ENTRY_Z:
                    state = 1  # long the spread
            elif abs(z_score) < PAIR_EXIT_Z:
                state = 0
            _pair_states[pair_number] = state

            # Normalize the two legs so neither pair contribution exceeds the
            # pair sleeve's per-asset dollar budget before sleeves are netted.
            pair_budget = PAIR_SLEEVE_WEIGHT * 10_000.0
            left_dollars = state * pair_budget / max(1.0, abs(hedge_ratio))
            right_dollars = -hedge_ratio * left_dollars
            pair_dollars[left_index - 1] += left_dollars
            pair_dollars[right_index - 1] += right_dollars

        synthetic_dollars = reversal_dollars + pair_dollars
        synthetic_dollars = np.clip(synthetic_dollars, -10_000.0, 10_000.0)

        # If the exact factor hedge would exceed ALGO's cap, scale the entire
        # synthetic book down proportionally before recomputing the hedge.
        aggregate_beta_dollars = float(betas @ synthetic_dollars)
        requested_hedge = HEDGE_FRACTION * aggregate_beta_dollars
        if abs(requested_hedge) > dollar_limits[0]:
            synthetic_dollars *= dollar_limits[0] / abs(requested_hedge)
            aggregate_beta_dollars = float(betas @ synthetic_dollars)

        target_dollars = np.zeros(n_instruments)
        target_dollars[1:] = synthetic_dollars
        target_dollars[0] = -HEDGE_FRACTION * aggregate_beta_dollars
        target_positions = np.rint(target_dollars / current_prices).astype(int)

    target_positions = np.clip(target_positions, -max_shares, max_shares).astype(int)

    # Apply the deadband after clipping, using a one-share minimum so positions
    # near zero do not churn. The held position is itself re-clipped to today's
    # moving share limit before it can be retained.
    held_positions = np.clip(_current_positions, -max_shares, max_shares).astype(int)
    thresholds = np.maximum(1, np.ceil(DEADBAND_FRACTION * np.abs(held_positions)).astype(int))
    keep_held = np.abs(target_positions - held_positions) <= thresholds
    target_positions = np.where(keep_held, held_positions, target_positions).astype(int)

    _current_positions = target_positions.copy()
    _last_num_days = n_days
    _last_output = target_positions.copy()
    return target_positions.astype(int)
