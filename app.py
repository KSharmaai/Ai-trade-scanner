#!/usr/bin/env python3
"""
app.py — Streamlit web UI for the AI Trade Scanner.
Run:  streamlit run app.py
"""

import os, json, datetime as dt
from pathlib import Path

import streamlit as st
import pandas as pd
import yaml

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Trade Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
CFG  = yaml.safe_load((ROOT / "config.yaml").read_text())

# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
market = st.sidebar.selectbox("Market", ["US", "IN"], index=0 if CFG["market"] == "US" else 1)
top_n  = st.sidebar.slider("Stocks to screen", 5, 15, 8)
use_screener = st.sidebar.toggle("Auto-screen (find best stocks)", value=True)

if not use_screener:
    custom = st.sidebar.text_area(
        "Manual watchlist (one per line)",
        value="\n".join(CFG["watchlist"]))
    manual_list = [t.strip() for t in custom.strip().split("\n") if t.strip()]

account  = st.sidebar.number_input("Paper account size", value=CFG["account_size"], step=1000)
risk_pct = st.sidebar.slider("Risk per trade %", 0.5, 3.0, CFG["risk_per_trade_pct"], 0.5)
min_rr   = st.sidebar.slider("Min R:R", 1.5, 4.0, CFG["min_risk_reward"], 0.5)

api_key = st.sidebar.text_input(
    "OpenAI API Key", type="password",
    value=os.environ.get("OPENAI_API_KEY", ""),
    help="Never committed — stays in your browser session only")

st.sidebar.markdown("---")
st.sidebar.caption("📌 Paper trading only · Not financial advice · You review, you decide")

# ── header ────────────────────────────────────────────────────────────────────
st.title("📈 AI Daily Trade Scanner")
st.caption(f"Today: {dt.date.today()} · Market: {market} · "
           f"Account: {account:,} · Risk/trade: {risk_pct}%")

col1, col2, col3 = st.columns([2,1,1])
with col1:
    run_btn = st.button("🔍 Run Scan Now", type="primary", use_container_width=True)
with col2:
    st.metric("Min R:R", f"{min_rr}:1")
with col3:
    st.metric("Max risk/trade", f"${account * risk_pct / 100:,.0f}")

st.divider()

# ── helpers ───────────────────────────────────────────────────────────────────
def confidence_badge(c: str) -> str:
    return {"high": "🟢 HIGH", "medium": "🟡 MEDIUM", "low": "🔴 LOW"}.get(c.lower(), c)

def status_badge(s: str) -> str:
    return {"APPROVED": "✅ APPROVED", "WATCHLIST": "👀 WATCHLIST",
            "REJECTED": "❌ REJECTED"}.get(s, s)

def run_scan(tickers: list, api_key: str) -> dict:
    import yfinance as yf
    import numpy as np
    from openai import OpenAI

    PIPELINE_PROMPT = """You are a cautious institutional trading analyst. PAPER TRADING ONLY.
Use ONLY the data below. Never invent prices. High-conviction only — returning zero setups is correct on most days.

For each setup, also flag: options_play (true/false) — whether this setup could work as an options trade
(needs ATR >2%, clear direction, 10+ days to expiry).

MARKET DATA ({today}):
{data_json}

Rules: min R:R {min_rr}:1. Confidence must be high or medium only — no low confidence setups.

Respond ONLY in valid JSON:
{{
  "scan": [{{"ticker":"...","trend":"up|down|sideways","note":"...","worth_watching":true}}],
  "setups": [{{
    "ticker":"...","type":"breakout|pullback|momentum|reversal",
    "direction":"long|short","evidence":"...","invalidation_reason":"...",
    "entry":0.0,"stop":0.0,"target":0.0,"timeframe_days":0,
    "confidence":"high|medium",
    "options_play":false,"options_note":"",
    "status":"APPROVED|WATCHLIST|REJECTED","status_reason":"..."
  }}],
  "market_note":"..."
}}"""

    # fetch data
    data = {}
    progress = st.progress(0, text="Fetching market data...")
    for i, ticker in enumerate(tickers):
        try:
            df = yf.download(ticker, period="120d", interval="1d",
                             progress=False, auto_adjust=True)
            if df is None or len(df) < 30:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
            delta = close.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rsi   = float((100 - 100 / (1 + gain / loss.replace(0, 1e-9))).iloc[-1])
            tr    = pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
            atr   = float(tr.rolling(14).mean().iloc[-1])
            data[ticker] = {
                "close": round(float(close.iloc[-1]), 2),
                "chg_1d_pct": round(float(close.pct_change().iloc[-1]*100), 2),
                "chg_5d_pct": round(float(close.pct_change(5).iloc[-1]*100), 2),
                "sma20": round(float(close.rolling(20).mean().iloc[-1]), 2),
                "sma50": round(float(close.rolling(50).mean().iloc[-1]), 2),
                "rsi14": round(rsi, 1),
                "atr14": round(atr, 2),
                "vol_vs_20d_avg": round(float(vol.iloc[-1]/max(vol.rolling(20).mean().iloc[-1],1)), 2),
                "high_20d": round(float(high.rolling(20).max().iloc[-1]), 2),
                "low_20d":  round(float(low.rolling(20).min().iloc[-1]), 2),
            }
        except Exception:
            pass
        progress.progress((i+1)/len(tickers), text=f"Fetched {ticker}...")

    progress.progress(1.0, text="Running AI analysis...")

    # call OpenAI
    client = OpenAI(api_key=api_key)
    import datetime as dt2
    response = client.chat.completions.create(
        model="gpt-4o", max_tokens=3000,
        messages=[{"role": "user", "content": PIPELINE_PROMPT.format(
            today=dt2.date.today(), data_json=json.dumps(data, indent=1),
            min_rr=min_rr)}])
    text = response.choices[0].message.content.strip()
    text = text.replace("```json","").replace("```","").strip()
    result = json.loads(text)

    # risk gate
    for s in result.get("setups", []):
        entry, stop, target = s["entry"], s["stop"], s["target"]
        per_share = abs(entry - stop)
        if per_share <= 0:
            s["status"] = "REJECTED"; s["status_reason"] = "invalid stop"
            s["shares"] = 0; s["dollar_risk"] = 0; s["rr"] = 0
            continue
        rr = abs(target - entry) / per_share
        shares = int((account * risk_pct / 100) // per_share)
        s["rr"] = round(rr, 2)
        s["shares"] = shares
        s["dollar_risk"] = round(shares * per_share, 2)
        if rr < min_rr:
            s["status"] = "REJECTED"
            s["status_reason"] = f"R:R {rr:.2f} below minimum {min_rr}"
        if shares == 0:
            s["status"] = "REJECTED"
            s["status_reason"] = "position too small at 1% risk"

    result["data"] = data
    progress.empty()
    return result


def save_memo(result: dict):
    today = dt.date.today().isoformat()
    Path("logs").mkdir(exist_ok=True)
    lines = [f"# Daily Decision Memo — {today}",
             f"*Market: {market} · Account: {account} · Risk: {risk_pct}%*", "",
             "**PAPER TRADING ONLY — HUMAN REVIEW REQUIRED**", "",
             "## Market read", result.get("market_note",""), "", "## Setups"]
    for s in result.get("setups", []):
        lines += [f"### {s['ticker']} {s['direction'].upper()} → {s['status']}",
                  f"- {s['evidence']}",
                  f"- Entry {s['entry']} · Stop {s['stop']} · Target {s['target']} · R:R {s.get('rr','?')}",
                  f"- Shares: {s.get('shares',0)} · Risk: ${s.get('dollar_risk',0)}",
                  f"- Options: {'Yes — ' + s.get('options_note','') if s.get('options_play') else 'No'}", ""]
    Path(f"logs/memo_{today}.md").write_text("\n".join(lines))


# ── main scan UI ──────────────────────────────────────────────────────────────
if run_btn:
    if not api_key:
        st.error("Enter your OpenAI API key in the sidebar first.")
        st.stop()

    with st.spinner("Screening stocks and running AI pipeline..."):
        if use_screener:
            from screener import get_top_candidates
            candidates = get_top_candidates(market, top_n)
            tickers = [c["ticker"] for c in candidates]

            st.subheader("🔎 Auto-screened candidates")
            df_screen = pd.DataFrame(candidates)[
                ["ticker","price","score","rsi","mom_5d","vol_ratio","above_sma20"]]
            df_screen.columns = ["Ticker","Price","Score","RSI","Mom 5d%","Vol Ratio","Above SMA20"]
            st.dataframe(df_screen, use_container_width=True, hide_index=True)
        else:
            tickers = manual_list

        result = run_scan(tickers, api_key)

    # market read
    st.subheader("🌍 Market Read")
    st.info(result.get("market_note", "No market note."))

    # scan table
    st.subheader("📊 Scan Results")
    scan_data = result.get("scan", [])
    if scan_data:
        df_scan = pd.DataFrame(scan_data)
        df_scan["worth_watching"] = df_scan["worth_watching"].map({True:"👀 Yes", False:"—"})
        st.dataframe(df_scan, use_container_width=True, hide_index=True)

    # setups
    st.subheader("🎯 Trade Setups")
    setups = result.get("setups", [])
    approved = [s for s in setups if s["status"] == "APPROVED"]
    watchlist = [s for s in setups if s["status"] == "WATCHLIST"]
    rejected  = [s for s in setups if s["status"] == "REJECTED"]

    if not setups:
        st.success("✅ No setups today — not trading is the right call.")
    else:
        for s in approved + watchlist + rejected:
            with st.expander(f"{status_badge(s['status'])} · {s['ticker']} "
                             f"{s['direction'].upper()} {s['type']} · "
                             f"{confidence_badge(s['confidence'])} · R:R {s.get('rr','?')}",
                             expanded=s["status"] == "APPROVED"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Entry", s["entry"])
                c2.metric("Stop", s["stop"])
                c3.metric("Target", s["target"])
                c4.metric("R:R", s.get("rr", "?"))
                c1.metric("Shares", s.get("shares", 0))
                c2.metric("$ at Risk", f"${s.get('dollar_risk',0):,}")
                c3.metric("Timeframe", f"{s.get('timeframe_days','?')}d")
                c4.metric("Options Play", "✅ Yes" if s.get("options_play") else "❌ No")

                st.markdown(f"**Evidence:** {s['evidence']}")
                st.markdown(f"**Invalidation:** {s['invalidation_reason']}")
                if s.get("options_play") and s.get("options_note"):
                    st.info(f"💡 Options: {s['options_note']}")
                st.caption(f"Status reason: {s['status_reason']}")

    # summary metrics
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stocks scanned", len(tickers))
    m2.metric("Setups found", len(setups))
    m3.metric("Approved", len(approved))
    m4.metric("Options plays", sum(1 for s in approved if s.get("options_play")))

    save_memo(result)
    st.caption(f"✅ Memo saved to logs/memo_{dt.date.today()}.md")

# ── past memos ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📁 Past Memos")
log_dir = Path("logs")
memos = sorted(log_dir.glob("memo_*.md"), reverse=True) if log_dir.exists() else []
if memos:
    selected = st.selectbox("Select date", [m.name for m in memos])
    if selected:
        st.markdown(Path(f"logs/{selected}").read_text())
else:
    st.caption("No memos yet — run your first scan above.")
