#!/usr/bin/env python3
"""
AI Daily Trade Scanner — paper trading decision-support pipeline.

Flow:  fetch data -> compute indicators -> Claude (scan + signals +
       trade plans) -> local risk gate (code, not AI) -> decision memo.

NOT financial advice. Paper trading only. Never auto-executes anything.
Usage:  export ANTHROPIC_API_KEY=sk-...   then   python daily_scan.py
"""

import os, sys, json, datetime as dt
from pathlib import Path

import yaml
import pandas as pd
import yfinance as yf
from anthropic import Anthropic

ROOT = Path(__file__).parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
TODAY = dt.date.today().isoformat()


# ---------------------------------------------------------------- data
def fetch(ticker: str) -> pd.DataFrame | None:
    df = yf.download(ticker, period=f"{CFG['lookback_days']}d",
                     interval=CFG["timeframe"], progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def indicators(df: pd.DataFrame) -> dict:
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    return {
        "close": round(float(close.iloc[-1]), 2),
        "chg_1d_pct": round(float(close.pct_change().iloc[-1] * 100), 2),
        "chg_5d_pct": round(float(close.pct_change(5).iloc[-1] * 100), 2),
        "sma20": round(float(close.rolling(20).mean().iloc[-1]), 2),
        "sma50": round(float(close.rolling(50).mean().iloc[-1]), 2),
        "rsi14": round(float(rsi.iloc[-1]), 1),
        "atr14": round(float(atr.iloc[-1]), 2),
        "vol_vs_20d_avg": round(float(vol.iloc[-1] / max(vol.rolling(20).mean().iloc[-1], 1)), 2),
        "high_20d": round(float(high.rolling(20).max().iloc[-1]), 2),
        "low_20d": round(float(low.rolling(20).min().iloc[-1]), 2),
    }


# ---------------------------------------------------------------- claude
PIPELINE_PROMPT = """You are a cautious trading analyst. PAPER TRADING ONLY, not financial advice.
Use ONLY the data below. Never invent prices, news, or certainty. No hype.

MARKET DATA (computed from real OHLCV, {today}):
{data_json}

RULES: min risk-to-reward {min_rr}:1. High-conviction setups only — it is GOOD to
return zero setups. Most days have nothing worth trading.

Do all stages and respond with ONLY valid JSON (no markdown fences):
{{
  "scan": [{{"ticker": "...", "trend": "up|down|sideways", "note": "...", "worth_watching": true}}],
  "setups": [{{
     "ticker": "...", "type": "breakout|pullback|momentum|reversal",
     "direction": "long|short", "evidence": "...", "invalidation_reason": "...",
     "entry": 0.0, "stop": 0.0, "target": 0.0, "timeframe_days": 0,
     "confidence": "high|medium",
     "status": "APPROVED|WATCHLIST|REJECTED", "status_reason": "..."
  }}],
  "market_note": "one-paragraph overall read, plainly stated, no predictions of certainty"
}}"""


def run_claudeAnt(data: dict) -> dict:
    client = Anthropic()  # reads ANTHROPIC_API_KEY
    msg = client.messages.create(
        model=CFG["model"], max_tokens=3000,
        messages=[{"role": "user", "content": PIPELINE_PROMPT.format(
            today=TODAY, data_json=json.dumps(data, indent=1),
            min_rr=CFG["min_risk_reward"])}])
    text = msg.content[0].text.strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(text)
       
def run_claude(data: dict) -> dict:
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=3000,
        messages=[{"role": "user", "content": PIPELINE_PROMPT.format(
            today=TODAY, data_json=json.dumps(data, indent=1),
            min_rr=CFG["min_risk_reward"])}])
    text = response.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)

# ---------------------------------------------------------------- risk gate (CODE, not AI)
def risk_gate(setup: dict) -> dict:
    """Deterministic checks. The AI proposes; this code disposes."""
    acct = CFG["account_size"]
    risk_cash = acct * CFG["risk_per_trade_pct"] / 100
    entry, stop, target = setup["entry"], setup["stop"], setup["target"]
    per_share = abs(entry - stop)
    if per_share <= 0:
        return {**setup, "status": "REJECTED", "status_reason": "invalid stop", "shares": 0}
    rr = abs(target - entry) / per_share
    shares = int(risk_cash // per_share)
    out = {**setup, "rr": round(rr, 2), "shares": shares,
           "dollar_risk": round(shares * per_share, 2)}
    if rr < CFG["min_risk_reward"]:
        out["status"] = "REJECTED"
        out["status_reason"] = f"R:R {rr:.2f} below minimum {CFG['min_risk_reward']}"
    if shares == 0:
        out["status"] = "REJECTED"
        out["status_reason"] = "position size rounds to zero at 1% risk"
    return out


# ---------------------------------------------------------------- memo
def write_memo(result: dict) -> Path:
    lines = [f"# Daily Decision Memo — {TODAY}",
             f"*Market: {CFG['market']} · Account (paper): {CFG['account_size']} · "
             f"Risk/trade: {CFG['risk_per_trade_pct']}%*",
             "", "**PAPER TRADING ONLY — HUMAN REVIEW REQUIRED. "
             "Educational, not financial advice. The AI did not act.**", "",
             "## Market read", result.get("market_note", "-"), "", "## Scan"]
    for s in result.get("scan", []):
        flag = "👀" if s.get("worth_watching") else "—"
        lines.append(f"- {flag} **{s['ticker']}** ({s['trend']}): {s['note']}")
    lines += ["", "## Setups"]
    setups = result.get("setups", [])
    if not setups:
        lines.append("No qualifying setups today. **Not trading is a valid outcome.**")
    for p in setups:
        lines += [f"### {p['ticker']} — {p['direction'].upper()} {p['type']} → **{p['status']}**",
                  f"- Evidence: {p['evidence']}",
                  f"- Entry {p['entry']} · Stop {p['stop']} · Target {p['target']} "
                  f"· R:R {p.get('rr','?')} · ~{p.get('timeframe_days','?')}d",
                  f"- Size @ {CFG['risk_per_trade_pct']}% risk: {p.get('shares',0)} shares "
                  f"(risking {p.get('dollar_risk',0)})",
                  f"- Invalidation: {p['invalidation_reason']}",
                  f"- Status reason: {p['status_reason']}", ""]
    path = ROOT / "logs" / f"memo_{TODAY}.md"
    path.write_text("\n".join(lines))
    return path


def append_trade_log(setups: list):
    log = ROOT / "logs" / "trade_log.csv"
    if not log.exists():
        log.write_text("date,ticker,direction,status,entry,stop,target,rr,shares,"
                       "outcome,exit_price,pnl,notes\n")
    with log.open("a") as f:
        for p in setups:
            f.write(f"{TODAY},{p['ticker']},{p['direction']},{p['status']},"
                    f"{p['entry']},{p['stop']},{p['target']},{p.get('rr','')},"
                    f"{p.get('shares','')},,,,\n")


# ---------------------------------------------------------------- main
def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first (keep it in .env, never in git).")
    print(f"Scanning {len(CFG['watchlist'])} tickers for {TODAY}...")
    data = {}
    for t in CFG["watchlist"]:
        df = fetch(t)
        if df is None or len(df) < 60:
            print(f"  ! {t}: no/insufficient data — skipped")
            continue
        data[t] = indicators(df)
        print(f"  ✓ {t}: close {data[t]['close']}, RSI {data[t]['rsi14']}")
    if not data:
        sys.exit("No data fetched. Check tickers / connectivity.")

    (ROOT / "data" / f"snapshot_{TODAY}.json").write_text(json.dumps(data, indent=2))
    print("Running Claude pipeline...")
    result = run_claude(data)
    result["setups"] = [risk_gate(s) for s in result.get("setups", [])]
    memo = write_memo(result)
    append_trade_log(result["setups"])
    print(f"\nDone → {memo}")
    for p in result["setups"]:
        print(f"  {p['status']:9} {p['ticker']} {p['direction']} "
              f"(R:R {p.get('rr','?')}, {p.get('shares',0)} sh)")
    print("\nReview the memo. YOU decide. Paper only.")


if __name__ == "__main__":
    main()
