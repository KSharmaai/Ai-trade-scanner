# AI Daily Trade Scanner

A plug-and-run daily pipeline: fetch real market data → compute indicators →
Claude proposes setups → **deterministic code enforces your risk rules** →
you get a decision memo. Works for **US and India (NSE)** tickers.

> Decision support, not financial advice. No trades are executed by this tool.
> You review, you decide, you own the risk.

## Setup (5 minutes)

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...        # or put in .env (gitignored)
# edit config.yaml → watchlist, account size, market
python daily_scan.py
```

India mode: set `market: IN` and use NSE tickers like `RELIANCE.NS`, `TCS.NS`,
`INFY.NS`, `NIFTYBEES.NS`. Everything else is identical.

## Output
- `logs/memo_YYYY-MM-DD.md` — the daily decision memo (APPROVED / WATCHLIST / REJECTED)
- `logs/trade_log.csv` — every setup logged; **you fill in outcome/exit/pnl**
- `data/snapshot_*.json` — the exact data Claude saw (auditability)

## Automating the "24/7" part
- **Linux/Mac cron** (weekdays, after market close):
  `30 16 * * 1-5 cd /path/ai-trade-scanner && python daily_scan.py`
- **Windows**: Task Scheduler → daily → `python daily_scan.py`
- **GitHub Actions** (free hosting): schedule a workflow with
  `cron: '30 21 * * 1-5'` (UTC) and store `ANTHROPIC_API_KEY` as a repo secret.

## The rules that matter more than the signals
1. **Risk gate is code, not AI.** Position size = 1% account risk / (entry−stop).
   R:R below 2:1 is auto-rejected. The AI cannot override this.
2. **Zero setups is a good day.** Overtrading kills accounts, not bad signals.
3. **50 logged trades before scaling up.** Then compute:
   win rate, average R, max drawdown from `trade_log.csv`. Those numbers —
   not hope — tell you if there's an edge.
4. Expectancy = (win% × avg win) − (loss% × avg loss). A 45% win rate with
   2.5R winners beats an 80% win rate with tiny wins and one blown stop.

## What this is NOT
- Not HFT (that requires co-location and microsecond infra — impossible here)
- Not a guarantee of any win rate
- Not connected to any broker
