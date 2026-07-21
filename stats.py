#!/usr/bin/env python3
"""
stats.py — reads logs/trade_log.csv and prints your real performance numbers.

Run after you've filled in the outcome/exit_price/pnl columns for closed trades.
Usage:  python stats.py
"""

import sys
from pathlib import Path
import pandas as pd

LOG = Path(__file__).parent / "logs" / "trade_log.csv"


def main():
    if not LOG.exists():
        sys.exit("No trade_log.csv yet — run daily_scan.py first, then log some outcomes.")

    df = pd.read_csv(LOG)

    # Only analyse closed trades (pnl filled in)
    closed = df[df["pnl"].notna() & (df["pnl"] != "")].copy()
    closed["pnl"] = pd.to_numeric(closed["pnl"], errors="coerce")
    closed = closed.dropna(subset=["pnl"])

    total = len(df)
    approved = len(df[df["status"] == "APPROVED"])
    closed_n = len(closed)

    print(f"\n{'='*50}")
    print(f"  AI Trade Scanner — Performance Stats")
    print(f"{'='*50}")
    print(f"  Total setups logged : {total}")
    print(f"  APPROVED plans      : {approved}")
    print(f"  Closed trades       : {closed_n}")

    if closed_n == 0:
        print("\n  No closed trades yet. Fill in pnl column in trade_log.csv.")
        return

    winners = closed[closed["pnl"] > 0]
    losers  = closed[closed["pnl"] <= 0]

    win_rate   = len(winners) / closed_n * 100
    avg_win    = winners["pnl"].mean() if len(winners) else 0
    avg_loss   = abs(losers["pnl"].mean()) if len(losers) else 0
    expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * avg_loss)
    total_pnl  = closed["pnl"].sum()

    # Max drawdown (running peak-to-trough)
    equity = closed["pnl"].cumsum()
    peak   = equity.cummax()
    dd     = equity - peak
    max_dd = dd.min()

    print(f"\n  Win rate            : {win_rate:.1f}%  ({len(winners)}W / {len(losers)}L)")
    print(f"  Avg winner          : +{avg_win:.2f}")
    print(f"  Avg loser           : -{avg_loss:.2f}")
    print(f"  Avg R:R (realised)  : {avg_win/avg_loss:.2f}" if avg_loss else "  Avg R:R            : N/A")
    print(f"  Expectancy/trade    : {expectancy:+.2f}")
    print(f"  Total P&L           : {total_pnl:+.2f}")
    print(f"  Max drawdown        : {max_dd:.2f}")
    print(f"{'='*50}\n")

    if closed_n < 30:
        print(f"  ⚠  Only {closed_n} closed trades — need 30+ for meaningful stats.")
    if expectancy > 0:
        print("  ✓ Positive expectancy — the system has a statistical edge so far.")
    else:
        print("  ✗ Negative expectancy — review setup criteria or risk rules.")

    print("\n  Past results do not guarantee future returns.\n")


if __name__ == "__main__":
    main()
