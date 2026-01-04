import os
import sys
import json
import time
import math
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Any

import yaml
import pyotp
import pandas as pd
import numpy as np
import logging
from kiteconnect import KiteTicker
import threading
import pathlib

from kite_trade import get_enctoken, KiteApp
from regime import classify_series, RANGE, TREND, EVENT
from backtester import vectorized_strangle, summary
from auth_utils import load_config, get_kite_with_refresh
from paper_broker import PaperBroker
import csv
from email_utils import send_email

CONFIG_PATH = os.getenv("TRADER_CONFIG", "config.yaml")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def fetch_historical(kite: KiteApp, instrument_token: int, start: dt.datetime, end: dt.datetime,
                     interval: str = "5minute", oi: bool = False) -> pd.DataFrame:
    data = kite.historical_data(instrument_token, start, end, interval, continuous=False, oi=oi)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def nearest_expiry(instruments: List[Dict[str, Any]], underlying: str) -> dt.date:
    today = dt.date.today()
    filtered = [i for i in instruments if i.get("name") == underlying and i.get("expiry") is not None]
    exp = sorted([i["expiry"] for i in filtered if i["expiry"] >= today])
    return exp[0] if exp else None


def get_live_ltp(kite: KiteApp, exchange: str, tradingsymbol: str) -> float:
    """Fetch live LTP; return 0 on any failure."""
    try:
        resp = kite.session.get(
            f"{kite.root_url}/quote/ltp",
            params={"i": f"{exchange}:{tradingsymbol}"},
            headers=kite.headers,
        ).json()
        return resp.get("data", {}).get(f"{exchange}:{tradingsymbol}", {}).get("last_price", 0)
    except Exception:
        return 0


def nearest_expiry_from_prefix(instruments: List[Dict[str, Any]], underlying: str) -> dt.date:
    today = dt.date.today()
    futs = [i for i in instruments if i.get("instrument_type") == "FUT" and str(i.get("tradingsymbol", "")).startswith(underlying)]
    exp = sorted({i["expiry"] for i in futs if i.get("expiry") and i["expiry"] >= today})
    return exp[0] if exp else None


INDEX_SYMBOLS = {
    "BANKNIFTY": "NSE:NIFTY BANK",
    "NIFTY": "NSE:NIFTY 50",
}


def get_index_tokens(nse_instruments: List[Dict[str, Any]]) -> Dict[str, int]:
    tokens = {}
    for inst in nse_instruments:
        name = str(inst.get("name", "")).upper()
        tradingsymbol = str(inst.get("tradingsymbol", "")).upper()
        if name in ("NIFTY BANK", "BANKNIFTY") or tradingsymbol in ("NIFTY BANK", "BANKNIFTY"):
            tokens["BANKNIFTY"] = inst.get("instrument_token")
        if name in ("NIFTY 50", "NIFTY") or tradingsymbol in ("NIFTY 50", "NIFTY"):
            tokens["NIFTY"] = inst.get("instrument_token")
    return tokens


def fetch_ltp_ws(token: int, api_key: str, enctoken: str, user_id: str, timeout: int = 5) -> float:
    """Fetch a single LTP via WebSocket for a given instrument_token."""
    result = {'ltp': 0.0}
    done = threading.Event()
    kws = KiteTicker(api_key=api_key, access_token=f"{enctoken}&user_id={user_id}")

    def on_ticks(ws, ticks):
        for t in ticks:
            if t.get('instrument_token') == token:
                result['ltp'] = t.get('last_price', 0.0) or 0.0
                done.set()
                ws.unsubscribe([token])
                ws.close()
                break

    def on_connect(ws, response):
        ws.subscribe([token])
        ws.set_mode(ws.MODE_QUOTE, [token])

    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    threading.Thread(target=kws.connect, kwargs={'threaded': True}, daemon=True).start()
    done.wait(timeout)
    try:
        kws.close()
    except Exception:
        pass
    return result['ltp']


def get_index_ltp(kite: KiteApp, underlying: str, override_symbol: str = None, nse_instruments: List[Dict[str, Any]] = None, enctoken: str = None, api_key: str = None, user_id: str = None) -> float:
    """Fetch index LTP via WebSocket token only (single best effort)."""
    if not (nse_instruments and api_key and enctoken and user_id):
        logger.warning("Index LTP WS missing prerequisites; falling back later")
        return 0
    token_map = get_index_tokens(nse_instruments)
    tok = token_map.get(underlying.upper())
    if not tok:
        logger.warning("No index token found in NSE instruments for %s", underlying)
        return 0
    ws_ltp = fetch_ltp_ws(tok, api_key, enctoken, user_id, timeout=5)
    logger.info("Index LTP via WS token %s => %.2f", tok, ws_ltp or 0)
    return ws_ltp or 0


def select_strikes(kite: KiteApp, instruments: List[Dict[str, Any]], underlying: str, wing_distance_pct: float,
                   index_symbol: str = None, nse_instruments: List[Dict[str, Any]] = None,
                   api_key: str = None, enctoken: str = None, user_id: str = None) -> Dict[str, Dict[str, Any]]:
    # First try index LTP
    spot = get_index_ltp(kite, underlying, index_symbol, nse_instruments, enctoken=enctoken,
                         api_key=api_key, user_id=user_id)
    if spot and spot > 0:
        logger.info("Spot from index LTP (WS) = %.2f", spot)
    else:
        logger.warning("Index LTP unavailable via WS; will fall back to futures/options")

    # Next try nearest future
    futs = [i for i in instruments if i.get("instrument_type") == "FUT" and str(i.get("tradingsymbol", "")).startswith(underlying)]
    futs = sorted(futs, key=lambda x: x.get("expiry", dt.date.max))
    exp = futs[0].get("expiry") if futs else nearest_expiry_from_prefix(instruments, underlying)

    if (not spot or spot <= 0) and futs:
        fut = futs[0]
        live = get_live_ltp(kite, fut.get("exchange", "NFO"), fut.get("tradingsymbol", ""))
        if live and live > 0:
            spot = live
            logger.info("Spot from FUT LTP %s = %.2f", fut.get("tradingsymbol"), spot)
        else:
            spot = fut.get("last_price")
            logger.info("Spot from FUT last_price %s = %.2f", fut.get("tradingsymbol"), spot)

    if not spot or spot <= 0:
        opt_strikes = sorted({i.get("strike") for i in instruments if i.get("instrument_type") in ("CE", "PE") and i.get("expiry") == exp and str(i.get("tradingsymbol", "")).startswith(underlying) and i.get("strike")})
        if opt_strikes:
            spot = opt_strikes[len(opt_strikes) // 2]  # midpoint strike as crude ATM
            logger.info("Spot from midpoint strikes (fallback) = %.2f", spot)

    if not spot or spot <= 0 or not exp:
        raise RuntimeError(f"Could not infer spot/expiry for {underlying}; check instruments or underlying symbol")

    def nearest(option_type: str, target: float):
        candidates = [i for i in instruments if i.get("instrument_type") == option_type and i.get("expiry") == exp and str(i.get("tradingsymbol", "")).startswith(underlying)]
        if not candidates:
            raise RuntimeError(f"No {option_type} options found for {underlying} expiry {exp}")
        return min(candidates, key=lambda x: abs(x.get("strike") - target))

    ce_target = spot * (1 + wing_distance_pct / 100)
    pe_target = spot * (1 - wing_distance_pct / 100)
    ce = nearest("CE", ce_target)
    pe = nearest("PE", pe_target)
    return {"ce": ce, "pe": pe, "expiry": exp, "spot": spot}


def paper_fill_price(ltp: float, spread: float, side: str) -> float:
    # Simulate realistic fill with partial spread consumption
    if side == "BUY":
        return ltp + spread * 0.3
    return ltp - spread * 0.3


def simulate_paper_trade(df: pd.DataFrame, entry_time: dt.time, exit_time: dt.time,
                         sl_pct: float, side: str = "SELL", slippage_ticks: float = 1.0,
                         tick_size: float = 0.05) -> Dict[str, Any]:
    # Simple intraday sim: enter at entry_time, exit at exit_time or SL hit.
    df = df.copy()
    df["time"] = df["date"].dt.time
    entry_bar = df[df["time"] >= entry_time].head(1)
    exit_bar = df[df["time"] >= exit_time].head(1)
    if entry_bar.empty or exit_bar.empty:
        return {"pnl": 0.0, "reason": "no_entry"}
    entry_price = entry_bar.iloc[0]["close"]
    stop_price = entry_price * (1 + sl_pct / 100) if side == "SELL" else entry_price * (1 - sl_pct / 100)
    pnl = 0.0
    hit_sl = False
    for _, row in df.iterrows():
        if row["time"] < entry_time:
            continue
        high = row["high"]
        low = row["low"]
        if side == "SELL" and high >= stop_price:
            fill = stop_price + slippage_ticks * tick_size
            pnl = entry_price - fill
            hit_sl = True
            break
        if side == "BUY" and low <= stop_price:
            fill = stop_price - slippage_ticks * tick_size
            pnl = fill - entry_price
            hit_sl = True
            break
        if row["time"] >= exit_time:
            exit_price = row["close"]
            pnl = (entry_price - exit_price) if side == "SELL" else (exit_price - entry_price)
            break
    return {"pnl": pnl, "hit_sl": hit_sl, "entry": entry_price, "stop": stop_price}


def compute_lots_for_risk(ce_entry: float, ce_stop: float, pe_entry: float, pe_stop: float,
                           lot_size: int, capital: float, per_trade_risk_pct: float) -> int:
    risk_per_unit = max(ce_stop - ce_entry, 0) + max(pe_stop - pe_entry, 0)
    risk_per_lot = risk_per_unit * lot_size
    allowed = capital * per_trade_risk_pct / 100 if per_trade_risk_pct else 0
    if risk_per_lot <= 0 or allowed <= 0:
        return 1
    lots = int(allowed // risk_per_lot)
    return max(lots, 1)


def simulate_paper_strangle(ce_df: pd.DataFrame, pe_df: pd.DataFrame, entry_time: dt.time, exit_time: dt.time,
                            sl_pct: float, lot_size: int, slippage_ticks: float, tick_size: float,
                            capital: float, per_trade_risk_pct: float,
                            trade_log_path: str = "paper_trades.csv") -> Dict[str, Any]:
    broker = PaperBroker(slippage_ticks=slippage_ticks, tick_size=tick_size, brokerage_per_order=40)
    ce_df = ce_df.copy(); pe_df = pe_df.copy()
    ce_df["time"] = ce_df["date"].dt.time
    pe_df["time"] = pe_df["date"].dt.time
    entry_ce = ce_df[ce_df["time"] >= entry_time].head(1)
    entry_pe = pe_df[pe_df["time"] >= entry_time].head(1)
    if entry_ce.empty or entry_pe.empty:
        return {"pnl": 0.0, "reason": "no_entry"}
    ce_entry_price = entry_ce.iloc[0]["close"]
    pe_entry_price = entry_pe.iloc[0]["close"]

    ce_stop = ce_entry_price * (1 + sl_pct / 100)
    pe_stop = pe_entry_price * (1 + sl_pct / 100)
    lots = compute_lots_for_risk(ce_entry_price, ce_stop, pe_entry_price, pe_stop,
                                 lot_size, capital, per_trade_risk_pct)
    qty = lot_size * lots
    broker.enter("CE", "SELL", ce_entry_price, qty, entry_ce.iloc[0]["date"])
    broker.enter("PE", "SELL", pe_entry_price, qty, entry_pe.iloc[0]["date"])

    hit_ce = False; hit_pe = False
    exit_ce_price = ce_entry_price; exit_pe_price = pe_entry_price

    for _, row in ce_df.iterrows():
        if row["time"] < entry_time:
            continue
        if row["high"] >= ce_stop and not hit_ce:
            exit_ce_price = ce_stop
            broker.exit("CE", "BUY", exit_ce_price, qty, row["date"])
            hit_ce = True
            break
    if not hit_ce:
        exit_row = ce_df[ce_df["time"] >= exit_time].head(1)
        if not exit_row.empty:
            exit_ce_price = exit_row.iloc[0]["close"]
            broker.exit("CE", "BUY", exit_ce_price, qty, exit_row.iloc[0]["date"])

    for _, row in pe_df.iterrows():
        if row["time"] < entry_time:
            continue
        if row["high"] >= pe_stop and not hit_pe:
            exit_pe_price = pe_stop
            broker.exit("PE", "BUY", exit_pe_price, qty, row["date"])
            hit_pe = True
            break
    if not hit_pe:
        exit_row = pe_df[pe_df["time"] >= exit_time].head(1)
        if not exit_row.empty:
            exit_pe_price = exit_row.iloc[0]["close"]
            broker.exit("PE", "BUY", exit_pe_price, qty, exit_row.iloc[0]["date"])

    pnl = broker.pnl()
    # write trades
    try:
        fieldnames = ["symbol", "side", "qty", "price", "timestamp", "type"]
        write_header = not os.path.exists(trade_log_path)
        with open(trade_log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            for t in broker.trades:
                w.writerow(t)
    except Exception as e:
        logger.warning("Failed to write trade log: %s", e)

    return {
        "pnl": pnl,
        "ce_entry": ce_entry_price,
        "pe_entry": pe_entry_price,
        "ce_exit": exit_ce_price,
        "pe_exit": exit_pe_price,
        "ce_hit_sl": hit_ce,
        "pe_hit_sl": hit_pe,
        "lots": lots,
        "qty": qty,
        "trades": broker.trades,
    }


def backtest_strangle(kite: KiteApp, instruments, strikes, cfg):
    lookback_days = cfg.get("backtest", {}).get("lookback_days", 15)
    interval = cfg.get("backtest", {}).get("interval", "5minute")
    trade_log_path = cfg.get("backtest", {}).get("trade_log_path")
    summary_path = cfg.get("backtest", {}).get("summary_path")
    equity_curve_path = cfg.get("backtest", {}).get("equity_curve_path")
    ce_path = cfg.get("backtest", {}).get("ce_path")
    pe_path = cfg.get("backtest", {}).get("pe_path")
    end = dt.datetime.now()
    start = end - dt.timedelta(days=lookback_days)

    if ce_path and pe_path:
        logger.info(f"Backtest: loading CE/PE from local files {ce_path}, {pe_path}")
        ce_df = load_local_candles(ce_path)
        pe_df = load_local_candles(pe_path)
    else:
        logger.info(f"Backtest: fetching CE/PE history for {lookback_days} days, interval={interval}")
        ce_df = fetch_historical(kite, strikes["ce"]["instrument_token"], start, end, interval, oi=True)
        pe_df = fetch_historical(kite, strikes["pe"]["instrument_token"], start, end, interval, oi=True)
    logger.info(f"Backtest: CE rows={len(ce_df)}, PE rows={len(pe_df)}")
    if ce_df.empty or pe_df.empty:
        logger.error("No historical data for backtest")
        return

    regimes = classify_series(ce_df)
    ce_dates = sorted({d.strftime('%Y-%m-%d') for d in ce_df['date'].dt.date.unique()})
    pe_dates = sorted({d.strftime('%Y-%m-%d') for d in pe_df['date'].dt.date.unique()})
    regime_counts = {}
    for r in regimes.values():
        regime_counts[r] = regime_counts.get(r, 0) + 1
    logger.info(f"CE days: {len(ce_dates)}, PE days: {len(pe_dates)}, common: {len(set(ce_dates).intersection(pe_dates))}")
    logger.info(f"Regime counts: {regime_counts}")

    allowed = cfg.get("backtest", {}).get("allowed_regimes", [RANGE, TREND, EVENT])
    logger.info(f"Allowed regimes: {allowed}")

    res = vectorized_strangle(
        ce_df,
        pe_df,
        regimes,
        entry_time=dt.time(9, 30),
        exit_time=dt.time(14, 45),
        sl_pct=cfg.get("risk", {}).get("leg_stop_pct", 25),
        brokerage_per_order=cfg.get("backtest", {}).get("brokerage_per_order", 40),
        lot_size=strikes['ce'].get('lot_size', 25),
        allowed_regimes=allowed,
        trade_log_path=trade_log_path,
        summary_path=summary_path,
        equity_curve_path=equity_curve_path,
    )
    logger.info(f"Backtest trades: {len(res)}")
    summary_data = summary(res)
    logger.info("Backtest summary: %s", json.dumps(summary_data, default=str, indent=2))
    maybe_send_alert(cfg, summary_data)


def maybe_send_alert(cfg: Dict[str, Any], summary_data: Dict[str, Any]):
    alerts_cfg = cfg.get("monitoring", {}).get("alerts", {}).get("email", {})
    if not alerts_cfg.get("enabled"):
        return
    to_addr = alerts_cfg.get("to")
    if not to_addr:
        logger.warning("Email alert enabled but no recipient configured")
        return
    to_list = [a.strip() for a in to_addr.split(",") if a.strip()]
    min_pnl = alerts_cfg.get("min_pnl_notify", 0)
    min_dd = alerts_cfg.get("min_drawdown_notify", 0)
    total = summary_data.get("total_pnl", 0)
    max_dd = summary_data.get("max_drawdown", 0)
    if (min_pnl and total >= min_pnl) or (min_dd and max_dd <= -abs(min_dd)):
        subject = f"Backtest Alert: PnL={total:.2f} DD={max_dd:.2f}"
        body = json.dumps(summary_data, indent=2, default=str)
        send_email(
            to_addrs=to_list,
            subject=subject,
            body=body,
            smtp_server=alerts_cfg.get("smtp_server"),
            smtp_port=alerts_cfg.get("smtp_port", 587),
            smtp_user=alerts_cfg.get("smtp_user"),
            smtp_password=alerts_cfg.get("smtp_password"),
            from_addr=alerts_cfg.get("from") or alerts_cfg.get("smtp_user"),
        )


def load_local_candles(path: str) -> pd.DataFrame:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Local data file not found: {path}")
    if p.suffix.lower() == '.parquet':
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df


def main(mode: str = "paper"):
    cfg = load_config()
    logger.info(f"Mode: {mode}")
    kite, enctoken = get_kite_with_refresh(cfg)
    profile = kite.profile()
    user_id = profile.get("user_id") if profile else None
    api_key = cfg.get("auth", {}).get("api_key")
    # Cache instruments
    instruments = kite.instruments("NFO")
    nse_instruments = kite.instruments("NSE")
    strat_cfg = cfg.get("strategy", {})
    underlying = strat_cfg.get("underlying", "BANKNIFTY")
    wing_pct = strat_cfg.get("wing_distance_pct", 1.0)
    index_symbol = strat_cfg.get("index_symbol")
    logger.info(f"Selecting strikes for {underlying} with wing_distance_pct={wing_pct}")
    strikes = select_strikes(kite, instruments, underlying, wing_pct, index_symbol, nse_instruments,
                             api_key=api_key, enctoken=enctoken, user_id=user_id)
    logger.info(f"Selected strikes: CE {strikes['ce']['tradingsymbol']} (lot {strikes['ce'].get('lot_size')}), PE {strikes['pe']['tradingsymbol']} (lot {strikes['pe'].get('lot_size')}), expiry {strikes['expiry']}, spot {strikes['spot']:.2f}")

    # For paper/backtest demo: pull recent data for CE leg
    lookback_days = cfg.get("backtest", {}).get("lookback_days", 5)
    interval = cfg.get("backtest", {}).get("interval", "5minute")
    logger.info(f"Fetching CE historical: token={strikes['ce']['instrument_token']} lookback_days={lookback_days} interval={interval}")
    end = dt.datetime.now()
    start = end - dt.timedelta(days=lookback_days)
    ce_df = fetch_historical(kite, strikes["ce"]["instrument_token"], start, end, interval, oi=True)
    pe_df = fetch_historical(kite, strikes["pe"]["instrument_token"], start, end, interval, oi=True)
    logger.info(f"CE candles fetched: {len(ce_df)} rows; PE candles: {len(pe_df)} rows")
    if ce_df.empty or pe_df.empty:
        logger.error("No historical data fetched; adjust lookback or symbol")
        sys.exit(1)
    res = simulate_paper_strangle(
        ce_df, pe_df,
        entry_time=dt.time(9, 30),
        exit_time=dt.time(14, 45),
        sl_pct=cfg.get("risk", {}).get("leg_stop_pct", 25),
        lot_size=strikes['ce'].get('lot_size', 25),
        slippage_ticks=cfg.get("execution", {}).get("slippage_ticks", 1.0),
        tick_size=strikes['ce'].get('tick_size', 0.05),
        capital=cfg.get("risk", {}).get("capital", 500000),
        per_trade_risk_pct=cfg.get("risk", {}).get("per_trade_risk_pct", 0.5),
        trade_log_path="paper_trades.csv",
    )
    logger.info(f"Paper strangle result: {res}")

    if mode == "backtest":
        backtest_strangle(kite, instruments, strikes, cfg)
        return

    if mode == "live":
        logger.warning("Live execution not implemented yet; this is the scaffold.")


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "paper"
    main(m)
