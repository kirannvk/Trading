# Trading System Scaffold (Zerodha Kite)

This adds a pro-trader oriented scaffold on top of `kite_trade.py` with TOTP login, basic strike selection, historical data fetch, and a paper-trade simulator. Extend it into full backtest/paper/live modes.

## Files
- `kite_trade.py` — Zerodha helper (as provided).
- `main.py` — entrypoint for paper/backtest scaffold.
- `requirements.txt` — dependencies.
- `config.example.yaml` — copy to `config.yaml` and fill secrets.

## Setup
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy config.example.yaml config.yaml  # fill user_id/password/totp_secret
```

## Run (paper scaffold demo)
```powershell
python main.py paper
```
The script:
- Logs in using TOTP (from config).
- Fetches instruments (NFO), picks nearest expiry strikes for the configured underlying.
- Pulls recent historical data for the CE leg.
- Simulates a simple intraday sell with SL/exit to illustrate the flow.

## Configuration (`config.yaml`)
Key sections:
- `auth`: `user_id`, `password`, `totp_secret` (Base32 TOTP secret).
- `risk`: capital, per-trade risk %, daily loss %, leg stop %, max concurrent.
- `session`: trading window and buffers.
- `execution`: slippage ticks, order type, exchange/product.
- `strategy`: underlying, wing distance %, rebalance/exit times.
- `backtest`: interval, lookback days, costs.
- `monitoring`: dashboard/alerts placeholders.

## Data collection (historical)
We ship a CLI collector to pull NIFTY/BANKNIFTY index and nearby options history into `data/`.

### Full daily history (index only) + 5y of 5m
```powershell
# BANKNIFTY daily from inception + 5 years of 5m, no options
python data_collector.py --underlying BANKNIFTY --interval 5minute --intraday_lookback_days 1825 --daily_start 1990-01-01 --skip_options --out_dir data

# NIFTY daily from inception + 5 years of 5m, no options
python data_collector.py --underlying NIFTY --interval 5minute --intraday_lookback_days 1825 --daily_start 1990-01-01 --skip_options --out_dir data
```

### Include nearest expiry options (ATM ± wing%)
```powershell
# BANKNIFTY index + weekly nearest expiry CE/PE, wing 1%
python data_collector.py --underlying BANKNIFTY --interval 5minute --lookback_days 365 --weekly --wing_pct 1.0 --out_dir data
```

Notes:
- Collector uses your `config.yaml` auth. It loads instruments from saved CSVs (nso_instruments.csv/nse_instruments.csv) or fetches live if missing.
- Output files are Parquet by default. To convert all Parquet in `data/` to CSV, run:
```powershell
python convert_parquet.py
```

## Dashboard
A simple Streamlit dashboard shows backtest trades/summary and paper trades.
```powershell
streamlit run dashboard.py
```
Environment variables to override defaults:
- `BACKTEST_TRADES` (default: backtest_trades.csv)
- `BACKTEST_SUMMARY` (default: backtest_trades_summary.json)
- `PAPER_TRADES` (default: paper_trades.csv)

## Backtest
Backtest writes trades, summary, and equity curve if configured in `config.yaml`:
- `backtest.trade_log_path` (e.g., backtest_trades.csv)
- `backtest.summary_path` (e.g., backtest_trades_summary.json)
- `backtest.equity_curve_path` (e.g., backtest_equity_curve.csv)

Run:
```powershell
python main.py backtest
```

## Paper mode
Runs strike selection, fetches CE/PE historicals, sizes by risk, and simulates a strangle with per-leg SL and time exit. Logs trades to `paper_trades.csv`.
```powershell
python main.py paper
```

## NIFTY short straddle paper trader (ws-driven)
Runs a hedged NIFTY ATM short straddle with 10% price wings, live WS ticks for SL/target, and logs to CSV.

```powershell
python nifty_short_straddle_papertrade.py
```
Behavior:
- Enforces entry at 09:30 and exit by 15:00 IST (skips and emails if past cutoff).
- Selects ATM CE/PE and 10% premium hedges; skips (with email) if no hedges found or spot/chain fetch fails.
- Uses WS quotes to monitor SL/target; falls back to polling on WS failure.
- Logs PnL and max intraday PnL; emails trade summary on exit.

Outputs:
- Log: `nifty_short_straddle_papertrade.log`
- CSV: `paper_trade_log.csv` with entry/exit, PnL, SL/target, symbols, hedges, and max_intraday_pnl.

Config options (`config.yaml`):
- `auth.api_key`, `auth.user_id/password/totp_secret` (token refresh happens every run).
- `monitoring.alerts.email`: set `enabled: true`, `to`, `from`, `smtp_server`, `smtp_port`, `smtp_user`, `smtp_password` (use an app password for Gmail). Used for exit and skip notifications.

Gmail SMTP example (app password):
```yaml
monitoring:
  alerts:
    email:
      enabled: true
      to: "you@gmail.com"
      from: "you@gmail.com"
      smtp_server: "smtp.gmail.com"
      smtp_port: 587
      smtp_user: "you@gmail.com"
      smtp_password: "<16-char app password>"
```
