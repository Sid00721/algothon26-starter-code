#!/usr/bin/env python3
"""Walk-forward diagnostics for the current teamName.py strategy."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd


def competition_score(mean_pnl: float, std_pnl: float) -> float:
    if mean_pnl <= 0 or std_pnl < 1e-10:
        return mean_pnl
    sharpe = np.sqrt(250) * mean_pnl / std_pnl
    return mean_pnl * sharpe**2 / (sharpe**2 + 1)


def backtest(prices: np.ndarray, start: int, end: int, position_function) -> dict[str, float]:
    n_instruments = prices.shape[0]
    position_limits = np.full(n_instruments, 10_000.0)
    position_limits[0] = 100_000.0
    commission_rates = np.full(n_instruments, 0.0001)
    commission_rates[0] = 0.00002

    position = np.zeros(n_instruments, dtype=int)
    pnl = []
    dollar_volume = 0.0

    for day in range(start, end):
        current_price = prices[:, day]
        requested = np.asarray(position_function(prices[:, : day + 1]))
        share_limits = (position_limits / current_price).astype(int)
        new_position = np.clip(requested, -share_limits, share_limits).astype(int)
        trade = new_position - position
        traded_dollars = current_price * np.abs(trade)
        commission = float(np.sum(traded_dollars * commission_rates))
        daily_pnl = float(new_position @ (prices[:, day + 1] - current_price) - commission)
        pnl.append(daily_pnl)
        dollar_volume += float(traded_dollars.sum())
        position = new_position

    pnl_array = np.asarray(pnl)
    mean_pnl = float(pnl_array.mean())
    std_pnl = float(pnl_array.std())
    sharpe = float(np.sqrt(250) * mean_pnl / std_pnl) if std_pnl else 0.0
    return {
        "start_day": start + 1,
        "end_day": end + 1,
        "days": end - start,
        "mean_pnl": mean_pnl,
        "std_pnl": std_pnl,
        "annualized_sharpe": sharpe,
        "score": competition_score(mean_pnl, std_pnl),
        "dollar_volume": dollar_volume,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", type=Path, default=Path("prices.txt"))
    parser.add_argument("--fold-size", type=int, default=50)
    parser.add_argument("--minimum-history", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("analysis/output/tables/walk_forward.csv"))
    args = parser.parse_args()

    price_frame = pd.read_csv(args.prices, sep=r"\s+")
    prices = price_frame.values.T
    strategy = importlib.import_module("teamName")

    records = []
    start = args.minimum_history
    final_trade_day = prices.shape[1] - 1
    while start < final_trade_day:
        end = min(start + args.fold_size, final_trade_day)
        records.append(backtest(prices, start, end, strategy.getMyPosition))
        start = end

    # eval.py first trades using the price at index (n_days - 250 - 1), then
    # scores the following 250 price changes through the final observation.
    records.append(backtest(prices, prices.shape[1] - 251, final_trade_day, strategy.getMyPosition))
    results = pd.DataFrame(records)
    results.index = [f"fold_{i + 1}" for i in range(len(records) - 1)] + ["official_visible_window"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index_label="window")
    print(results.to_string(float_format=lambda value: f"{value:,.2f}"))


if __name__ == "__main__":
    main()
