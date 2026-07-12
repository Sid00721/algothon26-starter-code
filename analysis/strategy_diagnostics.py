#!/usr/bin/env python3
"""Generate visible-window diagnostics for the current submission strategy."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teamName import getMyPosition


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis" / "output"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"


def savefig(path: Path) -> None:
    plt.savefig(path, dpi=155, bbox_inches="tight")
    plt.close()


def main() -> None:
    frame = pd.read_csv(ROOT / "prices.txt", sep=r"\s+")
    names = frame.columns
    prices = frame.values.T
    n_instruments, n_days = prices.shape
    caps = np.full(n_instruments, 10_000.0)
    caps[0] = 100_000.0
    fees = np.full(n_instruments, 0.0001)
    fees[0] = 0.00002

    start = n_days - 251
    end = n_days - 1
    position = np.zeros(n_instruments, dtype=int)
    records = []
    contributions = np.zeros(n_instruments)
    dollar_positions = []

    for day in range(start, end):
        current = prices[:, day]
        requested = np.asarray(getMyPosition(prices[:, : day + 1]))
        limits = (caps / current).astype(int)
        new_position = np.clip(requested, -limits, limits).astype(int)
        trade = new_position - position
        traded_dollars = current * np.abs(trade)
        commission_by_asset = traded_dollars * fees
        pnl_by_asset = new_position * (prices[:, day + 1] - current) - commission_by_asset
        dollar_position = new_position * current
        contributions += pnl_by_asset
        dollar_positions.append(dollar_position)
        records.append(
            {
                "day": day + 2,
                "gross_pnl": float((new_position * (prices[:, day + 1] - current)).sum()),
                "commission": float(commission_by_asset.sum()),
                "net_pnl": float(pnl_by_asset.sum()),
                "dollar_turnover": float(traded_dollars.sum()),
                "gross_exposure": float(np.abs(dollar_position).sum()),
                "net_exposure": float(dollar_position.sum()),
            }
        )
        position = new_position

    daily = pd.DataFrame(records).set_index("day")
    daily["cumulative_pnl"] = daily["net_pnl"].cumsum()
    daily["pnl_drawdown"] = daily["cumulative_pnl"] - daily["cumulative_pnl"].cummax()
    rolling_mean = daily["net_pnl"].rolling(30).mean()
    rolling_std = daily["net_pnl"].rolling(30).std()
    daily["rolling_30d_sharpe"] = np.sqrt(250) * rolling_mean / rolling_std
    daily.to_csv(TABLES / "strategy_daily_diagnostics.csv")

    contribution_frame = pd.DataFrame(
        {
            "net_pnl_contribution": contributions,
            "fraction_of_total": contributions / contributions.sum(),
        },
        index=names,
    ).sort_values("net_pnl_contribution")
    contribution_frame.to_csv(TABLES / "strategy_asset_contributions.csv")

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    axes[0].plot(daily.index, daily["cumulative_pnl"], linewidth=1.2)
    axes[0].fill_between(daily.index, daily["cumulative_pnl"], 0, alpha=0.12)
    axes[0].set_ylabel("Cumulative P&L ($)")
    axes[0].set_title("Current strategy: visible 250-day evaluation diagnostics")
    axes[1].bar(daily.index, daily["net_pnl"], width=1.0)
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_ylabel("Daily P&L ($)")
    axes[2].plot(daily.index, daily["rolling_30d_sharpe"], linewidth=1.0)
    axes[2].axhline(0, color="black", linewidth=0.7)
    axes[2].set_ylabel("30d ann. Sharpe")
    axes[3].plot(daily.index, daily["dollar_turnover"], label="Turnover", linewidth=0.9)
    axes[3].plot(daily.index, daily["gross_exposure"], label="Gross exposure", linewidth=0.9)
    axes[3].set_ylabel("Dollars")
    axes[3].set_xlabel("Day")
    axes[3].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    savefig(FIGURES / "16_strategy_diagnostics.png")

    colors = np.where(contribution_frame["net_pnl_contribution"] >= 0, "tab:blue", "tab:red")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(contribution_frame.index, contribution_frame["net_pnl_contribution"], color=colors)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(np.arange(len(contribution_frame)), contribution_frame.index, rotation=90, fontsize=7)
    ax.set_ylabel("Net P&L contribution ($)")
    ax.set_title("Current strategy P&L contribution by asset")
    savefig(FIGURES / "17_strategy_asset_contributions.png")

    exposure = np.asarray(dollar_positions).T / caps[:, None]
    fig, ax = plt.subplots(figsize=(15, 9))
    image = ax.imshow(exposure, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_yticks(np.arange(n_instruments), names, fontsize=6)
    ax.set_xlabel("Visible evaluation day")
    ax.set_title("Strategy positions as a fraction of each asset's limit")
    fig.colorbar(image, ax=ax, label="Fraction of position limit")
    savefig(FIGURES / "18_strategy_exposure_heatmap.png")


if __name__ == "__main__":
    main()
