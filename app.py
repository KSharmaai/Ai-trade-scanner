#!/usr/bin/env python3
"""
app.py — AI Trade Scanner · Professional UI (v4)
Run:  streamlit run app.py
Trading decision support. Not financial advice. Final decision is yours.
"""

import os, json, datetime as dt
from pathlib import Path

import streamlit as st
import pandas as pd
import yaml

st.set_page_config(page_title="AI Trade Scanner", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# ── styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1250px;}
  h1 {font-weight: 700; letter-spacing: -0.5px;}
  div[data-testid="stMetric"] {
      background: #f8f9fb; border: 1px solid #e6e8ee;
      border-radius: 10px; padding: 12px 16px;}
  div[data-testid="stMetricLabel"] {font-size: 0.78rem; color: #6b7280;}
  .setup-approved {border-left: 5px solid #16a34a; padding-left: 12px;}
  .setup-watch    {border-left: 5px solid #f59e0b; padding-left: 12px;}
  .setup-rejected {border-left: 5px solid #dc2626; padding-left: 12px;}
  .score-pill {display:inline-block; padding:4px 14px; border-radius:20px;
      font-weight:700; font-size:1.05rem;}
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).parent
CFG  = yaml.safe_load((ROOT / "config.yaml").read_text())

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    market = st.selectbox("Market", ["US", "IN"],
                          help="US = NYSE/Nasdaq stocks · IN = India NSE stocks")
    top_n  = st.slider("Stocks to screen", 5, 15, 8,
                       help="How many top-ranked stocks the auto-screener picks")
    use_screener = st.toggle("Auto-screen (find best stocks)", value=True,
                             help="ON: system picks the most active/trending stocks automatically. OFF: use your own list.")
    if not use_screener:
        custom = st.text_area("Manual watchlist (one per line)",
                              value="\n".join(CFG["watchlist"]))
        manual_list = [t.strip() for t in custom.strip().split("\n") if t.strip()]

    st.divider()
    account  = st.number_input("Account size", value=CFG["account_size"], step=1000,
                               help="Capital base used to calculate position sizes")
    risk_pct = st.slider("Risk per trade %", 0.5, 3.0, CFG["risk_per_trade_pct"], 0.5,
                         help="Max % of your account you can lose if ONE trade hits its stop-loss. Professionals use 1-2%.")
    min_rr   = st.slider("Minimum Risk:Reward", 1.5, 4.0, CFG["min_risk_reward"], 0.5,
                         help="Reward must be at least this many times the risk. 2:1 means: risk $100 to potentially make $200.")
    api_key = st.text_input("OpenAI API Key", type="password",
                            value=os.environ.get("OPENAI_API_KEY", ""))
    st.divider()
    st.caption("Decision support · Not financial advice · Final decision is yours")

# ── header ────────────────────────────────────────────────────────────────────
st.title("📈 AI Trade Scanner")
st.caption(f"{dt.date.today():%A, %B %d, %Y} · {market} market · "
           f"Account {account:,} · {risk_pct}% risk per trade")

with st.expander("📖 New here? Every term explained in plain English"):
    st.markdown("""
| Term | Meaning |
|------|---------|
| **Entry** | The price at which the plan says to buy (or sell short) |
| **Stop-loss (Stop)** | The exit price if the trade goes WRONG. Caps your loss. Non-negotiable. |
| **Target** | The exit price if the trade goes RIGHT. Where you take profit. |
| **Risk:Reward (R:R)** | Potential profit ÷ potential loss. R:R 2.0 = risking $1 to make $2. Higher is better. |
| **RSI** | Momentum meter 0–100. Above 70 = possibly overbought (expensive). Below 30 = possibly oversold (cheap). |
| **SMA20 / SMA50** | Average price over last 20/50 days. Price above them = uptrend. |
| **ATR** | Average daily price movement. Higher = more volatile = bigger swings both ways. |
| **Trade Score /10** | Overall setup quality: combines R:R, trend alignment, momentum and confidence. 7+ is strong. |
| **Invalidation** | The condition that proves the trade idea wrong — exit if it happens. |
| **APPROVED / WATCHLIST / REJECTED** | Passed all rules / interesting but wait / failed a rule (usually R:R too low) |

**The golden rule:** a high Trade Score is NOT a guarantee. It means the setup follows good rules. Roughly half of good setups still lose — the math works because winners are 2x+ bigger than losers.
""")

c1, c2, c3 = st.columns([2, 1, 1])
run_btn = c1.button("🔍 Run Scan Now", type="primary", use_container_width=True)
c2.metric("Min R:R", f"{min_rr}:1", help="Plans below this ratio get auto-rejected")
c3.metric("Max loss per trade", f"${account*risk_pct/100:,.0f}",
          help="The most you can lose on one trade if the stop-loss triggers")
st.divider()


# ── scoring & helpers ─────────────────────────────────────────────────────────
def trade_score(s: dict, ind: dict) -> float:
    """Score setup quality 0-10. Transparent formula, no black box."""
    score = 0.0
    rr = s.get("rr", 0)
    score += min(4.0, rr * 1.5)                                   # R:R up to 4 pts
    score += 2.0 if s.get("confidence") == "high" else 1.0        # AI confidence
    long = s["direction"] == "long"
    above20 = ind["close"] > ind["sma20"]; above50 = ind["close"] > ind["sma50"]
    if (long and above20 and above50) or (not long and not above20 and not above50):
        score += 2.0                                              # trend alignment
    if 40 <= ind["rsi14"] <= 70:
        score += 1.0                                              # RSI not extreme
    if ind["vol_vs_20d_avg"] > 1.0:
        score += 1.0                                              # volume backing
    return round(min(10.0, score), 1)


def score_color(x: float) -> str:
    return "#16a34a" if x >= 7 else "#f59e0b" if x >= 5 else "#dc2626"


def setup_chart(ticker: str, df: pd.DataFrame, s: dict):
    """Candlestick chart with entry/stop/target zones + win/lose scenario paths."""
    import plotly.graph_objects as go
    d = df.tail(60)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"],
                                 low=d["Low"], close=d["Close"], name=ticker,
                                 increasing_line_color="#16a34a",
                                 decreasing_line_color="#dc2626"))
    sma20 = df["Close"].rolling(20).mean().tail(60)
    fig.add_trace(go.Scatter(x=d.index, y=sma20, name="SMA20",
                             line=dict(color="#3b82f6", width=1.5)))
    entry, stop, target = s["entry"], s["stop"], s["target"]
    last_date = d.index[-1]
    horizon = s.get("timeframe_days", 10)
    future = pd.bdate_range(last_date, periods=horizon + 1)[1:]

    # zones
    fig.add_hrect(y0=min(entry, target), y1=max(entry, target),
                  fillcolor="#16a34a", opacity=0.08, line_width=0)
    fig.add_hrect(y0=min(entry, stop), y1=max(entry, stop),
                  fillcolor="#dc2626", opacity=0.08, line_width=0)
    for y, name, color in [(entry, "Entry", "#3b82f6"),
                           (target, "Target", "#16a34a"),
                           (stop, "Stop-loss", "#dc2626")]:
        fig.add_hline(y=y, line_dash="dot", line_color=color,
                      annotation_text=f"{name} {y}", annotation_position="right")

    # scenario paths (NOT forecasts — just the plan visualised)
    if len(future) > 0:
        fig.add_trace(go.Scatter(x=[last_date, future[-1]], y=[entry, target],
                                 mode="lines", name="If WIN (plan)",
                                 line=dict(color="#16a34a", dash="dash", width=2)))
        fig.add_trace(go.Scatter(x=[last_date, future[-1]], y=[entry, stop],
                                 mode="lines", name="If LOSS (plan)",
                                 line=dict(color="#dc2626", dash="dash", width=2)))

    fig.update_layout(height=420, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.08),
                      title=f"{ticker} — last 60 days + trade plan")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("⚠️ Dashed lines are the PLAN visualised (win vs loss scenario) — "
               "not a price prediction. Nobody can predict the path.")


def run_scan(tickers: list, api_key: str) -> dict:
    import yfinance as yf
    from openai import OpenAI

    PROMPT = """You are a cautious institutional trading analyst providing decision support.
Use ONLY the data given. Never invent prices. High-conviction setups only —
zero setups is the correct answer on most days.
Also flag options_play (true/false): could this work as an options trade
(needs ATR% > 2, clear direction, 10+ day horizon)? Keep options_note short.

MARKET DATA ({today}):
{data_json}

Rules: min R:R {min_rr}:1. Confidence: high or medium only.
Respond ONLY in valid JSON:
{{"scan":[{{"ticker":"...","trend":"up|down|sideways","note":"...","worth_watching":true}}],
"setups":[{{"ticker":"...","type":"breakout|pullback|momentum|reversal","direction":"long|short",
"evidence":"...","invalidation_reason":"...","entry":0.0,"stop":0.0,"target":0.0,
"timeframe_days":0,"confidence":"high|medium","options_play":false,"options_note":"",
"status":"APPROVED|WATCHLIST|REJECTED","status_reason":"..."}}],
"market_note":"..."}}"""

    data, frames = {}, {}
    prog = st.progress(0, text="Fetching market data...")
    for i, t in enumerate(tickers):
        try:
            df = yf.download(t, period="120d", interval="1d",
                             progress=False, auto_adjust=True)
            if df is None or len(df) < 30:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
            delta = close.diff()
            rsi = 100 - 100/(1 + delta.clip(lower=0).rolling(14).mean() /
                             (-delta.clip(upper=0)).rolling(14).mean().replace(0, 1e-9))
            tr  = pd.concat([high-low, (high-close.shift()).abs(),
                             (low-close.shift()).abs()], axis=1).max(axis=1)
            data[t] = {
                "close": round(float(close.iloc[-1]), 2),
                "chg_1d_pct": round(float(close.pct_change().iloc[-1]*100), 2),
                "chg_5d_pct": round(float(close.pct_change(5).iloc[-1]*100), 2),
                "sma20": round(float(close.rolling(20).mean().iloc[-1]), 2),
                "sma50": round(float(close.rolling(50).mean().iloc[-1]), 2),
                "rsi14": round(float(rsi.iloc[-1]), 1),
                "atr14": round(float(tr.rolling(14).mean().iloc[-1]), 2),
                "vol_vs_20d_avg": round(float(vol.iloc[-1]/max(vol.rolling(20).mean().iloc[-1], 1)), 2),
                "high_20d": round(float(high.rolling(20).max().iloc[-1]), 2),
                "low_20d": round(float(low.rolling(20).min().iloc[-1]), 2),
            }
            frames[t] = df
        except Exception:
            pass
        prog.progress((i+1)/len(tickers), text=f"Fetched {t}")

    prog.progress(1.0, text="Running AI analysis...")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o", max_tokens=3000,
        messages=[{"role": "user", "content": PROMPT.format(
            today=dt.date.today(), data_json=json.dumps(data, indent=1), min_rr=min_rr)}])
    text = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
    result = json.loads(text)

    # deterministic risk gate + trade score
    for s in result.get("setups", []):
        per_share = abs(s["entry"] - s["stop"])
        if per_share <= 0:
            s.update(status="REJECTED", status_reason="invalid stop", rr=0,
                     shares=0, dollar_risk=0, score=0)
            continue
        rr = abs(s["target"] - s["entry"]) / per_share
        shares = int((account * risk_pct / 100) // per_share)
        s.update(rr=round(rr, 2), shares=shares,
                 dollar_risk=round(shares * per_share, 2))
        if rr < min_rr:
            s.update(status="REJECTED",
                     status_reason=f"R:R {rr:.2f} below your {min_rr} minimum")
        if shares == 0:
            s.update(status="REJECTED",
                     status_reason="position rounds to 0 shares at this risk level")
        s["score"] = trade_score(s, data.get(s["ticker"], {"close":0,"sma20":0,"sma50":0,"rsi14":50,"vol_vs_20d_avg":0}))
    result["data"], result["frames"] = data, frames
    prog.empty()
    return result


def save_memo(result: dict):
    today = dt.date.today().isoformat()
    Path("logs").mkdir(exist_ok=True)
    lines = [f"# Daily Decision Memo — {today}",
             f"*Market: {market} · Account: {account} · Risk: {risk_pct}%*", "",
             "## Market read", result.get("market_note", ""), "", "## Setups"]
    for s in result.get("setups", []):
        lines += [f"### {s['ticker']} {s['direction'].upper()} → {s['status']} "
                  f"(Score {s.get('score','?')}/10)",
                  f"- {s['evidence']}",
                  f"- Entry {s['entry']} · Stop {s['stop']} · Target {s['target']} "
                  f"· R:R {s.get('rr','?')} · {s.get('shares',0)} shares "
                  f"· risk ${s.get('dollar_risk',0)}", ""]
    Path(f"logs/memo_{today}.md").write_text("\n".join(lines))


# ── main flow ─────────────────────────────────────────────────────────────────
if run_btn:
    if not api_key:
        st.error("Enter your OpenAI API key in the sidebar first."); st.stop()

    if use_screener:
        from screener import get_top_candidates
        with st.spinner(f"Screening the {market} universe for today's most active stocks..."):
            candidates = get_top_candidates(market, top_n)
        tickers = [c["ticker"] for c in candidates]
        st.subheader("🔎 Today's screened candidates")
        st.caption("The system ranked the whole universe and picked these — based on "
                   "momentum, volume surge, volatility and trend. Higher score = more active/interesting.")
        dfc = pd.DataFrame(candidates)[["ticker","price","score","rsi","mom_5d","vol_ratio","above_sma20"]]
        dfc.columns = ["Ticker","Price","Screen Score","RSI","5-day Move %","Volume vs Avg","In Uptrend"]
        st.dataframe(dfc, use_container_width=True, hide_index=True)
    else:
        tickers = manual_list

    result = run_scan(tickers, api_key)
    st.session_state["result"] = result

if "result" in st.session_state:
    result = st.session_state["result"]

    st.subheader("🌍 Market read")
    st.info(result.get("market_note", "No note."))

    setups   = result.get("setups", [])
    approved = [s for s in setups if s["status"] == "APPROVED"]
    watch    = [s for s in setups if s["status"] == "WATCHLIST"]
    rejected = [s for s in setups if s["status"] == "REJECTED"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stocks analysed", len(result.get("data", {})))
    m2.metric("✅ Approved", len(approved), help="Passed every rule — ready for your review")
    m3.metric("👀 Watchlist", len(watch), help="Interesting but needs confirmation first")
    m4.metric("❌ Rejected", len(rejected), help="Broke a rule — usually reward too small vs risk")

    st.subheader("🎯 Trade setups")
    if not setups:
        st.success("**No setups today.** The system found nothing that passes the quality bar. "
                   "Not trading is a professional decision — cash is a position too.")
    for s in approved + watch + rejected:
        color = score_color(s.get("score", 0))
        badge = {"APPROVED":"✅","WATCHLIST":"👀","REJECTED":"❌"}[s["status"]]
        with st.expander(
                f"{badge} {s['ticker']} · {s['direction'].upper()} {s['type']} · "
                f"Trade Score {s.get('score','?')}/10 · R:R {s.get('rr','?')}",
                expanded=(s["status"] == "APPROVED")):

            st.markdown(
                f"<span class='score-pill' style='background:{color}22;color:{color};'>"
                f"Trade Score: {s.get('score','?')}/10</span> &nbsp; "
                f"<b>{s['status']}</b> — {s['status_reason']}",
                unsafe_allow_html=True)
            st.caption("Score = R:R quality (max 4) + trend alignment (2) + AI confidence (2) "
                       "+ healthy RSI (1) + volume backing (1). It rates rule-following, not certainty.")

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Entry", s["entry"], help="Price to enter the trade")
            k2.metric("Stop-loss", s["stop"], help="Exit here if wrong — caps the loss")
            k3.metric("Target", s["target"], help="Exit here if right — takes the profit")
            k4.metric("Risk:Reward", s.get("rr","?"), help="Profit potential ÷ loss potential")
            k1.metric("Shares", s.get("shares",0), help="Position size at your risk % setting")
            k2.metric("Money at risk", f"${s.get('dollar_risk',0):,}",
                      help="Actual $ lost if stop-loss hits")
            k3.metric("Time horizon", f"{s.get('timeframe_days','?')} days")
            k4.metric("Options candidate", "Yes ✅" if s.get("options_play") else "No",
                      help="Whether this setup could also be played with options")

            st.markdown(f"**Why this setup:** {s['evidence']}")
            st.markdown(f"**Idea is wrong if:** {s['invalidation_reason']}")
            if s.get("options_play") and s.get("options_note"):
                st.info(f"💡 Options angle: {s['options_note']}")

            frames = result.get("frames", {})
            if s["ticker"] in frames:
                setup_chart(s["ticker"], frames[s["ticker"]], s)

    save_memo(result)
    st.caption(f"💾 Memo saved: logs/memo_{dt.date.today()}.md")

# ── past memos ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📁 Past memos")
memos = sorted(Path("logs").glob("memo_*.md"), reverse=True) if Path("logs").exists() else []
if memos:
    sel = st.selectbox("Select date", [m.name for m in memos])
    st.markdown(Path(f"logs/{sel}").read_text())
else:
    st.caption("No memos yet — run your first scan.")
