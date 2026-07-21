#!/usr/bin/env python3
"""
screener.py — automatically finds tradeable stocks instead of a fixed list.
Screens for: high volume, momentum, volatility — like institutional quant desks do.
Returns top US and India (NSE) candidates for the daily scan.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime

# ── Curated universe (institutional-grade liquid stocks) ──────────────────────
# These are high-liquidity names institutions actually trade.
# We screen WITHIN this universe — not random penny stocks.

US_UNIVERSE = [
    # Mega cap tech (high liquidity, options available)
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    # Finance
    "JPM", "GS", "BAC",
    # ETFs (market pulse)
    "SPY", "QQQ", "IWM",
    # Semiconductors
    "AMD", "INTC", "AVGO",
    # Other momentum names
    "NFLX", "CRM", "UBER",
]

INDIA_UNIVERSE = [
    # Nifty 50 blue chips (NSE tickers need .NS suffix)
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BAJFINANCE.NS", "KOTAKBANK.NS", "LT.NS",
    # ETFs
    "NIFTYBEES.NS", "BANKBEES.NS",
]


def score_stock(ticker: str, lookback: int = 60) -> dict | None:
    """Download data and compute a momentum/volatility score."""
    try:
        df = yf.download(ticker, period=f"{lookback}d", interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or len(df) < 30:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df["Close"]
        vol   = df["Volume"]

        # Indicators
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1] if len(df) >= 50 else sma20
        price = float(close.iloc[-1])
        atr   = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]
        atr_pct = float(atr / price * 100)

        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = float(100 - 100 / (1 + gain / loss.replace(0, 1e-9)).iloc[-1])

        vol_ratio = float(vol.iloc[-1] / max(vol.rolling(20).mean().iloc[-1], 1))
        mom_5d    = float(close.pct_change(5).iloc[-1] * 100)
        mom_20d   = float(close.pct_change(20).iloc[-1] * 100)
        above_sma20 = price > float(sma20)
        above_sma50 = price > float(sma50)

        # Scoring (0-100) — higher = more interesting to scan
        score = 0
        score += min(30, abs(mom_5d) * 3)          # recent momentum
        score += min(20, vol_ratio * 10)            # volume surge
        score += min(20, atr_pct * 4)               # volatility (opportunity)
        score += 15 if above_sma20 else 0           # trend filter
        score += 15 if 40 < rsi < 75 else 0         # RSI sweet spot (not extreme)

        return {
            "ticker": ticker,
            "price": round(price, 2),
            "rsi": round(rsi, 1),
            "atr_pct": round(atr_pct, 2),
            "vol_ratio": round(vol_ratio, 2),
            "mom_5d": round(mom_5d, 2),
            "mom_20d": round(mom_20d, 2),
            "above_sma20": above_sma20,
            "above_sma50": above_sma50,
            "score": round(score, 1),
        }
    except Exception:
        return None


def get_top_candidates(market: str = "US", top_n: int = 8) -> list[dict]:
    """Screen universe and return top N candidates by score."""
    universe = US_UNIVERSE if market == "US" else INDIA_UNIVERSE
    print(f"  Screening {len(universe)} {market} stocks...")
    results = []
    for t in universe:
        r = score_stock(t)
        if r:
            results.append(r)
    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:top_n]
    print(f"  Top {len(top)} candidates: {[r['ticker'] for r in top]}")
    return top


if __name__ == "__main__":
    print("=== US Top Candidates ===")
    for s in get_top_candidates("US", 5):
        print(f"  {s['ticker']:12} score={s['score']:5.1f}  RSI={s['rsi']}  "
              f"mom5d={s['mom_5d']}%  vol_ratio={s['vol_ratio']}")
    print("\n=== India Top Candidates ===")
    for s in get_top_candidates("IN", 5):
        print(f"  {s['ticker']:20} score={s['score']:5.1f}  RSI={s['rsi']}  "
              f"mom5d={s['mom_5d']}%  vol_ratio={s['vol_ratio']}")
