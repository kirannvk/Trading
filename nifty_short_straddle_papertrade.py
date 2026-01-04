import datetime as dt
import time
import pandas as pd
import logging
import threading
import sys
from zoneinfo import ZoneInfo

# Set IST timezone
IST = ZoneInfo("Asia/Kolkata")

from auth_utils import load_config, get_kite_with_refresh
from instrument_cache import InstrumentCache
from option_chain import fetch_option_chain, fetch_option_quotes_ws, fetch_spot_ws, fetch_quotes
from kiteconnect import KiteTicker
from email_utils import send_email

ENTRY_TIME = dt.time(9, 30)
EXIT_TIME = dt.time(15, 0)
STOP_LOSS_PCT = 0.30
TARGET_PCT = 0.50
LOG_FILE = 'paper_trade_log.csv'

# Feature flags to control WS usage
USE_WS_FOR_SPOT = True
USE_WS_FOR_CHAIN_QUOTES = True
USE_WS_MONITOR = True

# Basic logger for this script
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers.clear()
_fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler = logging.FileHandler('nifty_short_straddle_papertrade.log', mode='a')
file_handler.setFormatter(_fmt)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setFormatter(_fmt)
logger.addHandler(console_handler)


def get_atm_option_tokens(df, spot):
    # Pick strike nearest to provided spot
    atm_strike = int(df.loc[(df['strike'] - spot).abs().idxmin(), 'strike'])

    def _pick(row):
        ltp = row.get('ltp')
        if pd.isna(ltp) or ltp is None:
            bid = row.get('bid')
            ask = row.get('ask')
            if bid is not None and ask is not None:
                return (bid + ask) / 2.0
            return bid or ask or 0.0
        return ltp

    ce = df[(df['strike'] == atm_strike) & (df['type'] == 'CE')].iloc[0]
    pe = df[(df['strike'] == atm_strike) & (df['type'] == 'PE')].iloc[0]
    lot_size = int(ce.get('lot_size', 1) or 1)
    return atm_strike, int(ce['token']), int(pe['token']), _pick(ce), _pick(pe), lot_size


def _fetch_live_quotes(kite, api_key, enctoken, user_id, tokens, symbols, last_prices):
    # Try WS first with retries, then REST fallback, then keep last prices
    for attempt in range(3):
        try:
            data = fetch_option_quotes_ws(api_key, enctoken, user_id, tokens)
            if data:
                return data
        except Exception as e:
            logging.warning("WS quote attempt %s failed: %s", attempt + 1, e)
        time.sleep(1)
    # REST fallback
    try:
        q = fetch_quotes(kite, symbols)
        out = {}
        for sym, tok in zip(symbols, tokens):
            dq = q.get(sym) or q.get(sym.replace('NFO:', '')) or {}
            out[tok] = dq
        if out:
            return out
    except Exception as e:
        logging.warning("REST quote fallback failed: %s", e)
    # fallback to last seen prices
    out = {}
    for tok in tokens:
        out[tok] = {'last_price': last_prices.get(tok, 0.0)}
    return out


def run_straddle_ws(api_key, enctoken, user_id, ce_token, pe_token, ce_entry, pe_entry, ce_sl, pe_sl, target, exit_time,
                    lot_size, hedge_ce_token=None, hedge_ce_entry=None, hedge_pe_token=None, hedge_pe_entry=None):
    """Single WS session: subscribe once, evaluate on each tick, close when done."""
    logging.info(
        "WS monitor start | tokens CE %s PE %s | entry CE %.2f PE %.2f | SL CE %.2f PE %.2f | target %.2f | lot_size %s",
        ce_token, pe_token, ce_entry, pe_entry, ce_sl, pe_sl, target, lot_size
    )
    result = {
        'ce_exit_price': None,
        'pe_exit_price': None,
        'ce_exit': None,
        'pe_exit': None,
        'hedge_ce_exit_price': None,
        'hedge_pe_exit_price': None,
        'hedge_ce_exit': None,
        'hedge_pe_exit': None,
        'max_pnl': 0.0,
    }
    last_prices = {ce_token: ce_entry, pe_token: pe_entry}
    if hedge_ce_token and hedge_pe_token:
        last_prices[hedge_ce_token] = hedge_ce_entry if hedge_ce_entry is not None else ce_entry * 0.1
        last_prices[hedge_pe_token] = hedge_pe_entry if hedge_pe_entry is not None else pe_entry * 0.1
    stop_event = threading.Event()
    tokens = [int(ce_token), int(pe_token)]
    if hedge_ce_token and hedge_pe_token:
        tokens += [int(hedge_ce_token), int(hedge_pe_token)]
    kws = KiteTicker(api_key=api_key, access_token=f"{enctoken}&user_id={user_id}")

    def close_ws():
        try:
            kws.unsubscribe(tokens)
        except Exception:
            pass
        try:
            kws.close()
        except Exception:
            pass

    def evaluate_now():
        nonlocal ce_sl, pe_sl
        ce_price = last_prices[ce_token]
        pe_price = last_prices[pe_token]
        current_pnl = ((ce_entry - (result['ce_exit_price'] or ce_price)) + (
                    pe_entry - (result['pe_exit_price'] or pe_price))) * lot_size
        if hedge_ce_token and hedge_pe_token:
            hedge_ce_price = last_prices[hedge_ce_token]
            hedge_pe_price = last_prices[hedge_pe_token]
            current_pnl += ((hedge_ce_price - hedge_ce_entry) + (hedge_pe_price - hedge_pe_entry)) * lot_size
        if current_pnl > result['max_pnl']:
            result['max_pnl'] = current_pnl
        loss_amount = ((ce_sl - ce_entry) + (pe_sl - pe_entry)) * lot_size
        logging.info(
            "WS tick CE %.2f PE %.2f | SL CE %.2f PE %.2f | Target %.2f | LossAmt %.2f | MaxPnL %.2f | PnL %.2f",
            ce_price, pe_price, ce_sl, pe_sl, target, loss_amount, result['max_pnl'], current_pnl
        )

        # If either SL hits, close both legs immediately at current prices
        if result['ce_exit'] is None and ce_price >= ce_sl:
            result['ce_exit'] = 'SL'
            result['ce_exit_price'] = ce_price
            result['pe_exit_price'] = result['pe_exit_price'] or pe_price
            logging.info("WS: CE SL hit at %.2f, closing both legs", ce_price)
            stop_event.set()
            close_ws()
            return
        if result['pe_exit'] is None and pe_price >= pe_sl:
            result['pe_exit'] = 'SL'
            result['pe_exit_price'] = pe_price
            result['ce_exit_price'] = result['ce_exit_price'] or ce_price
            logging.info("WS: PE SL hit at %.2f, closing both legs", pe_price)
            stop_event.set()
            close_ws()
            return

        # Target check: if combined PnL meets/exceeds target, exit both
        if current_pnl >= target:
            result['ce_exit_price'] = result['ce_exit_price'] or ce_price
            result['pe_exit_price'] = result['pe_exit_price'] or pe_price
            logging.info("WS: target hit, CE %.2f PE %.2f", result['ce_exit_price'], result['pe_exit_price'])
            stop_event.set()
            close_ws()
            return

    def on_ticks(ws, ticks):
        logging.info("WS: received %d ticks", len(ticks))
        for t in ticks:
            tok = t.get('instrument_token')
            if tok in tokens:
                lp = t.get('last_price', last_prices.get(tok, 0.0)) or 0.0
                last_prices[tok] = lp
                logging.info("WS tick token=%s ltp=%.2f", tok, lp)
        if not stop_event.is_set():
            evaluate_now()

    def on_connect(ws, response):
        logging.info("WS: connected, subscribing tokens %s", tokens)
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_QUOTE, tokens)

    def on_error(ws, code, reason):
        logging.warning("run_straddle_ws error code=%s reason=%s", code, reason)
        stop_event.set()
        close_ws()

    def on_close(ws, code, reason):
        logging.warning("run_straddle_ws closed code=%s reason=%s", code, reason)
        stop_event.set()
        close_ws()

    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    kws.on_error = on_error
    kws.on_close = on_close

    # Timer to force exit at cutoff
    def force_exit():
        if not stop_event.is_set():
            result['ce_exit_price'] = result['ce_exit_price'] or last_prices[ce_token]
            result['pe_exit_price'] = result['pe_exit_price'] or last_prices[pe_token]
            logging.info("WS: cutoff reached, CE %.2f PE %.2f", result['ce_exit_price'], result['pe_exit_price'])
            stop_event.set()
            close_ws()

    now_dt = dt.datetime.now(IST)
    exit_dt = dt.datetime.combine(now_dt.date(), exit_time).replace(tzinfo=IST)  # Make exit_dt timezone-aware
    if exit_dt < now_dt:
        exit_dt = exit_dt + dt.timedelta(days=1)
    delay = (exit_dt - now_dt).total_seconds()
    logging.info("WS: scheduling cutoff in %.1f seconds at %s", delay, exit_dt)
    threading.Timer(delay, force_exit).start()

    logging.info("WS: starting connect (threaded=False)")
    # Start WS (non-threaded connect as requested)
    kws.connect(threaded=False)
    logging.info("WS: connect() returned, stop_event=%s", stop_event.is_set())
    stop_event.wait()
    logging.info(
        "WS monitor done | ce_exit_price=%s pe_exit_price=%s ce_exit=%s pe_exit=%s max_pnl=%.2f",
        result['ce_exit_price'], result['pe_exit_price'], result['ce_exit'], result['pe_exit'], result['max_pnl']
    )
    close_ws()
    return result


def get_next_expiry(ic: InstrumentCache, underlying: str, today: dt.date):
    """Return the earliest option expiry strictly after today for the given underlying."""
    df = ic.nfo[ic.nfo['tradingsymbol'].str.startswith(underlying)] if hasattr(ic, 'nfo') else pd.DataFrame()
    if df.empty or 'expiry' not in df.columns:
        logging.error("get_next_expiry: no instruments for %s or missing expiry column", underlying)
        raise RuntimeError("No option contracts found for underlying")
    df = df.copy()
    df['expiry'] = pd.to_datetime(df['expiry']).dt.date
    # Prefer OPTIDX if present, else allow CE/PE rows
    opt_df = df[df['instrument_type'].isin(['OPTIDX'])]
    if opt_df.empty:
        opt_df = df[df['instrument_type'].isin(['CE', 'PE'])]
    if opt_df.empty:
        logging.error("get_next_expiry: no OPTIDX/CE/PE rows for %s", underlying)
        raise RuntimeError("No option contracts found for underlying")
    future_expiries = opt_df[opt_df['expiry'] > today]['expiry'].drop_duplicates().sort_values()
    if future_expiries.empty:
        logging.error("get_next_expiry: no future expiries after %s for %s", today, underlying)
        raise RuntimeError("No future expiries available for underlying")
    return future_expiries.iloc[0]


def select_hedge_option(df: pd.DataFrame, atm_strike: int, opt_type: str, target_price: float):
    """Pick OTM hedge of same type with LTP nearest to target_price."""
    if opt_type == 'CE':
        candidates = df[(df['type'] == 'CE') & (df['strike'] > atm_strike)].copy()
    else:
        candidates = df[(df['type'] == 'PE') & (df['strike'] < atm_strike)].copy()
    candidates = candidates[candidates['ltp'].notna()]
    if candidates.empty:
        return None
    candidates['dist'] = (candidates['ltp'] - target_price).abs()
    pick = candidates.sort_values(['dist', 'strike']).iloc[0]
    return int(pick['token']), float(pick['ltp']), int(pick['strike'])


def _send_trade_summary_email(cfg, trade_row):
    alerts = cfg.get('monitoring', {}).get('alerts', {}).get('email', {}) or {}
    if not alerts.get('enabled'):
        return
    to_addrs = alerts.get('to') or []
    if isinstance(to_addrs, str):
        to_addrs = [a.strip() for a in to_addrs.split(',') if a.strip()]
    if not to_addrs:
        logging.warning("Email alerts enabled but no recipients configured")
        return
    subject = f"NIFTY paper straddle exit | PnL {trade_row['profit']:.2f}"
    lines = [
        f"Date: {trade_row['date']}",
        f"Entry: {trade_row['entry_time']}",
        f"Exit: {trade_row['exit_time']}",
        f"Strike: {trade_row['strike']}",
        f"Qty: {trade_row['lot_size']}",
        f"PnL: {trade_row['profit']:.2f}",
        f"Max intraday PnL: {trade_row.get('max_intraday_pnl', 0):.2f}",
        f"Target: {trade_row['target_amount']:.2f}",
        f"Loss amount: {trade_row['loss_amount']:.2f}",
        f"CE {trade_row['call_symbol']} entry {trade_row['call_entry']:.2f} exit {trade_row['call_exit']:.2f}",
        f"PE {trade_row['put_symbol']} entry {trade_row['put_entry']:.2f} exit {trade_row['put_exit']:.2f}",
        f"Hedge CE {trade_row['hedge_call_symbol']} entry {trade_row['hedge_ce_entry']:.2f} exit {trade_row['hedge_ce_exit']:.2f}",
        f"Hedge PE {trade_row['hedge_put_symbol']} entry {trade_row['hedge_pe_entry']:.2f} exit {trade_row['hedge_pe_exit']:.2f}",
    ]
    body = "\n".join(lines)
    err = send_email(
        to_addrs=to_addrs,
        subject=subject,
        body=body,
        smtp_server=alerts.get('smtp_server', ''),
        smtp_port=int(alerts.get('smtp_port', 587)),
        smtp_user=alerts.get('smtp_user', ''),
        smtp_password=alerts.get('smtp_password', ''),
        from_addr=alerts.get('from') or alerts.get('smtp_user') or None,
    )
    if err:
        logging.warning("Failed to send trade summary email: %s", err)
    else:
        logging.info("Trade summary email sent to %s", to_addrs)


def _send_skip_notice_email(cfg, reason: str):
    alerts = cfg.get('monitoring', {}).get('alerts', {}).get('email', {}) or {}
    if not alerts.get('enabled'):
        return
    to_addrs = alerts.get('to') or []
    if isinstance(to_addrs, str):
        to_addrs = [a.strip() for a in to_addrs.split(',') if a.strip()]
    if not to_addrs:
        logging.warning("Email alerts enabled but no recipients configured")
        return
    subject = f"NIFTY paper straddle skipped"
    lines = [
        f"Reason: {reason}",
        f"Time: {dt.datetime.now(IST)}",  # Always use IST
    ]
    body = "\n".join(lines)
    err = send_email(
        to_addrs=to_addrs,
        subject=subject,
        body=body,
        smtp_server=alerts.get('smtp_server', ''),
        smtp_port=int(alerts.get('smtp_port', 587)),
        smtp_user=alerts.get('smtp_user', ''),
        smtp_password=alerts.get('smtp_password', ''),
        from_addr=alerts.get('from') or alerts.get('smtp_user') or None,
    )
    if err:
        logging.warning("Failed to send skip notice email: %s", err)
    else:
        logging.info("Skip notice email sent to %s", to_addrs)


def _is_market_closed(cfg, today: dt.date) -> str:
    # Weekend check
    if today.weekday() >= 5:
        return "Market closed (weekend)"
    # Holiday file check
    cal_path = cfg.get('market_calendar_file') if isinstance(cfg, dict) else None
    cal_path = cal_path or 'nse_holidays.csv'
    if cal_path and pd.io.common.file_exists(cal_path):
        try:
            hol = pd.read_csv(cal_path)
            date_col = None
            for c in hol.columns:
                if str(c).lower() in ('date', 'holiday_date'):
                    date_col = c
                    break
            if date_col:
                hol_dates = pd.to_datetime(hol[date_col]).dt.date.dropna().tolist()
                if today in hol_dates:
                    return f"Market closed (holiday {today})"
        except Exception as e:
            logging.warning("Failed reading holiday calendar %s: %s", cal_path, e)
    return ""


def place_order(kite, symbol, token, qty, order_type, price=None, real_trade=False):
    """Abstract order placement for both paper and real trades. Always use NFO for options."""
    if not real_trade:
        response = {"order_id": "paper", "status": "success"}
        logging.info(f"Paper trade: {order_type} {qty} of {symbol} at {price} | response: {response}")
        return response
    try:
        # Always use NFO for options
        order_id = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_NFO,
            tradingsymbol=symbol,
            transaction_type=order_type,
            quantity=qty,
            product=kite.PRODUCT_MIS,
            order_type=kite.ORDER_TYPE_MARKET if price is None else kite.ORDER_TYPE_LIMIT,
            price=price,
            validity=None,
            disclosed_quantity=None,
            trigger_price=None,
            squareoff=None,
            stoploss=None,
            trailing_stoploss=None,
            tag="AlgoTrader"
        )
        if order_id is None:
            logging.error(f"Kite returned None for order_id. Check authentication/session and API status. Params: symbol={symbol}, qty={qty}, order_type={order_type}, price={price}")
            response = {"order_id": None, "status": "failed", "error": "Kite returned None for order_id. Check authentication/session and API status."}
            return response
        response = {"order_id": order_id, "status": "success"}
        logging.info(f"Real trade: {order_type} {qty} of {symbol} at {price}, order_id={order_id} | response: {response}")
        return response
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        response = {"order_id": None, "status": "failed", "error": str(e), "traceback": tb}
        logging.error(f"Order failed: {e} | response: {response}")
        return response


def paper_trade():
    logging.info("Starting paper trade session for NIFTY short straddle")
    cfg = load_config()
    trade_mode = cfg.get("trade_mode", "paper")
    real_trade = trade_mode == "real"
    logging.info(f"Trade mode from config: {trade_mode} | real_trade={real_trade}")
    today = dt.datetime.now(IST).date()  # Always use IST
    closed_reason = _is_market_closed(cfg, today)
    if closed_reason:
        reason = f"{closed_reason}; skipping trade"
        logging.info(reason)
        _send_skip_notice_email(cfg, reason)
        return
    now_time = dt.datetime.now(IST).time()  # Always use IST
    if now_time >= EXIT_TIME:
        reason = f"Current time {now_time} is past cutoff {EXIT_TIME}; skipping trade"
        logging.info(reason)
        _send_skip_notice_email(cfg, reason)
        return
    kite, enctoken = get_kite_with_refresh(cfg)
    ic = InstrumentCache.from_files_or_api(kite)
    api_key = cfg.get('auth', {}).get('api_key')
    profile = kite.profile()
    user_id = profile.get('user_id') if profile else None

    # No KiteConnect margin client in use
    kc = None

    logging.info("Waiting for entry time %s", ENTRY_TIME)
    while dt.datetime.now(IST).time() < ENTRY_TIME:  # Always use IST
        time.sleep(5)

    # Fetch spot via WS first (optional), then futures fallback
    tok = ic.get_index_token('NIFTY') if hasattr(ic, 'get_index_token') else None
    spot = 0
    if USE_WS_FOR_SPOT and api_key and enctoken and user_id and tok:
        spot = fetch_spot_ws(api_key, enctoken, user_id, tok, timeout=6)
        logging.info("Spot via WS token %s = %.2f", tok, spot)
    if not spot or spot <= 0:
        futs = ic.nfo[(ic.nfo['instrument_type'] == 'FUT') & (ic.nfo['tradingsymbol'].str.startswith('NIFTY'))]
        if not futs.empty:
            fut = futs.sort_values('expiry').iloc[0]
            spot = fut.get('last_price', 0) or 0
            logging.info("Spot via futures fallback %s = %.2f", fut.get('tradingsymbol'), spot)
    if not spot or spot <= 0:
        reason = "Cannot fetch spot via WS or futures; aborting"
        logging.error(reason)
        _send_skip_notice_email(cfg, reason)
        return

    try:
        logging.info("Fetching option chain for NIFTY strikes_around_atm=5 with spot %.2f", spot)
        expiry = get_next_expiry(ic, 'NIFTY', dt.datetime.now(IST).date())  # Always use IST
        logging.info("Selected expiry %s (next available after today)", expiry)
        chain_api_key = api_key if USE_WS_FOR_CHAIN_QUOTES else None
        chain_enctoken = enctoken if USE_WS_FOR_CHAIN_QUOTES else None
        chain_user = user_id if USE_WS_FOR_CHAIN_QUOTES else None
        spot, expiry, df = fetch_option_chain(
            kite, ic, 'NIFTY', strikes_around_atm=5, spot=spot, expiry=expiry, api_key=chain_api_key,
            enctoken=chain_enctoken, user_id=chain_user
        )
        logging.info("Fetched option chain. Spot=%s Expiry=%s rows=%s", spot, expiry, len(df))
    except Exception as e:
        logging.exception("Failed to fetch option chain / spot: %s", e)
        _send_skip_notice_email(cfg, f"Failed to fetch option chain / spot: {e}")
        return

    atm_strike, ce_token, pe_token, ce_entry, pe_entry, lot_size = get_atm_option_tokens(df, spot)
    atm_strike = int(atm_strike)
    ce_hedge = select_hedge_option(df, atm_strike, 'CE', ce_entry * 0.10)
    pe_hedge = select_hedge_option(df, atm_strike, 'PE', pe_entry * 0.10)
    if not ce_hedge or not pe_hedge:
        reason = "Could not find hedge options close to 10% price; aborting"
        logging.error(reason)
        _send_skip_notice_email(cfg, reason)
        return
    hedge_ce_token, hedge_ce_entry, hedge_ce_strike = ce_hedge
    hedge_pe_token, hedge_pe_entry, hedge_pe_strike = pe_hedge
    symbols = [
        f"NFO:{df.loc[df['token'] == ce_token, 'symbol'].iloc[0]}",
        f"NFO:{df.loc[df['token'] == pe_token, 'symbol'].iloc[0]}",
        f"NFO:{df.loc[df['token'] == hedge_ce_token, 'symbol'].iloc[0]}",
        f"NFO:{df.loc[df['token'] == hedge_pe_token, 'symbol'].iloc[0]}"
    ]
    ce_symbol = df.loc[df['token'] == ce_token, 'symbol'].iloc[0]
    pe_symbol = df.loc[df['token'] == pe_token, 'symbol'].iloc[0]
    hedge_ce_symbol = df.loc[df['token'] == hedge_ce_token, 'symbol'].iloc[0]
    hedge_pe_symbol = df.loc[df['token'] == hedge_pe_token, 'symbol'].iloc[0]
    logging.info(
        "ATM %s | CE short %s (%s) @ %.2f | PE short %s (%s) @ %.2f | lot_size %s | Hedge CE %s (%s) @ %.2f | Hedge PE %s (%s) @ %.2f",
        atm_strike, ce_token, ce_symbol, ce_entry, pe_token, pe_symbol, pe_entry, lot_size,
        hedge_ce_token, hedge_ce_symbol, hedge_ce_entry, hedge_pe_token, hedge_pe_symbol, hedge_pe_entry
    )

    ce_sl = ce_entry * (1 + STOP_LOSS_PCT)
    pe_sl = pe_entry * (1 + STOP_LOSS_PCT)
    target = (ce_entry + pe_entry) * TARGET_PCT * lot_size
    loss_amount = ((ce_sl - ce_entry) + (pe_sl - pe_entry)) * lot_size

    ce_exit = pe_exit = None
    ce_exit_price = pe_exit_price = None
    hedge_ce_exit_price = hedge_pe_exit_price = None
    entry_ts = dt.datetime.now(IST)  # Always use IST
    exit_ts = None
    last_prices = {
        ce_token: ce_entry,
        pe_token: pe_entry,
        hedge_ce_token: hedge_ce_entry,
        hedge_pe_token: hedge_pe_entry,
    }
    max_pnl = 0.0

    # Place entry orders for all legs
    logging.info("Placing entry orders for all legs...")
    place_order(kite, ce_symbol, ce_token, lot_size, "SELL", ce_entry, real_trade)
    place_order(kite, pe_symbol, pe_token, lot_size, "SELL", pe_entry, real_trade)
    place_order(kite, hedge_ce_symbol, hedge_ce_token, lot_size, "BUY", hedge_ce_entry, real_trade)
    place_order(kite, hedge_pe_symbol, hedge_pe_token, lot_size, "BUY", hedge_pe_entry, real_trade)

    # Use WS streaming to monitor; fallback to polling if WS fails or disabled
    if USE_WS_MONITOR:
        try:
            end_time = EXIT_TIME
            logging.info("Starting WS monitor for CE %s PE %s", ce_token, pe_token)
            ws_result = run_straddle_ws(
                api_key, enctoken, user_id,
                ce_token, pe_token, ce_entry, pe_entry, ce_sl, pe_sl, target, end_time, lot_size,
                hedge_ce_token, hedge_ce_entry, hedge_pe_token, hedge_pe_entry
            )
            ce_exit_price = ws_result.get('ce_exit_price') or ce_exit_price or last_prices[ce_token]
            pe_exit_price = ws_result.get('pe_exit_price') or pe_exit_price or last_prices[pe_token]
            hedge_ce_exit_price = ws_result.get('hedge_ce_exit_price') or hedge_ce_exit_price or last_prices[
                hedge_ce_token]
            hedge_pe_exit_price = ws_result.get('hedge_pe_exit_price') or hedge_pe_exit_price or last_prices[
                hedge_pe_token]
            max_pnl = ws_result.get('max_pnl', max_pnl)
            exit_ts = dt.datetime.now(IST)  # Always use IST
        except Exception as e:
            logging.warning("WS monitor failed, fallback to polling: %s", e)
            while True:
                now = dt.datetime.now(IST).time()  # Always use IST
                if now >= EXIT_TIME:
                    live = _fetch_live_quotes(kite, api_key, enctoken, user_id,
                                              [ce_token, pe_token, hedge_ce_token, hedge_pe_token], symbols,
                                              last_prices)
                    ce_exit_price = ce_exit_price or live.get(ce_token, {}).get('last_price', ce_entry)
                    pe_exit_price = pe_exit_price or live.get(pe_token, {}).get('last_price', pe_entry)
                    hedge_ce_exit_price = hedge_ce_exit_price or live.get(hedge_ce_token, {}).get('last_price',
                                                                                                  hedge_ce_entry)
                    hedge_pe_exit_price = hedge_pe_exit_price or live.get(hedge_pe_token, {}).get('last_price',
                                                                                                  hedge_pe_entry)
                    logging.info("Exit time reached. CE exit %.2f PE exit %.2f HedgeCE %.2f HedgePE %.2f",
                                 ce_exit_price, pe_exit_price, hedge_ce_exit_price, hedge_pe_exit_price)
                    exit_ts = dt.datetime.now(IST)  # Always use IST
                    break

                live = _fetch_live_quotes(kite, api_key, enctoken, user_id,
                                          [ce_token, pe_token, hedge_ce_token, hedge_pe_token], symbols, last_prices)
                ce_price = live.get(ce_token, {}).get('last_price', last_prices.get(ce_token, ce_entry))
                pe_price = live.get(pe_token, {}).get('last_price', last_prices.get(pe_token, pe_entry))
                hedge_ce_price = live.get(hedge_ce_token, {}).get('last_price',
                                                                  last_prices.get(hedge_ce_token, hedge_ce_entry))
                hedge_pe_price = live.get(hedge_pe_token, {}).get('last_price',
                                                                  last_prices.get(hedge_pe_token, hedge_pe_entry))
                last_prices[ce_token] = ce_price
                last_prices[pe_token] = pe_price
                last_prices[hedge_ce_token] = hedge_ce_price
                last_prices[hedge_pe_token] = hedge_pe_price
                logging.info(
                    "Live CE %.2f PE %.2f | HedgeCE %.2f HedgePE %.2f | SL CE %.2f PE %.2f | Target %.2f | LossAmt %.2f",
                    ce_price, pe_price, hedge_ce_price, hedge_pe_price, ce_sl, pe_sl, target, loss_amount)

                if not ce_exit and ce_price >= ce_sl:
                    ce_exit = 'SL'
                    ce_exit_price = ce_price
                    pe_sl = pe_price * (1 + STOP_LOSS_PCT)
                    hedge_ce_exit_price = hedge_ce_price
                    hedge_pe_exit_price = hedge_pe_price
                    exit_ts = dt.datetime.now(IST)  # Always use IST
                    logging.info("CE SL hit at %.2f. Closing hedges CE %.2f PE %.2f", ce_exit_price,
                                 hedge_ce_exit_price, hedge_pe_exit_price)
                if not pe_exit and pe_price >= pe_sl:
                    pe_exit = 'SL'
                    pe_exit_price = pe_price
                    ce_sl = ce_price * (1 + STOP_LOSS_PCT)
                    hedge_ce_exit_price = hedge_ce_price
                    hedge_pe_exit_price = hedge_pe_exit_price
                    exit_ts = dt.datetime.now(IST)  # Always use IST
                    logging.info("PE SL hit at %.2f. Closing hedges CE %.2f PE %.2f", pe_exit_price,
                                 hedge_ce_exit_price, hedge_pe_exit_price)

                total_pnl = ((ce_entry - (ce_exit_price or ce_price)) + (pe_entry - (pe_exit_price or pe_price)) + (
                            hedge_ce_price - hedge_ce_entry) + (hedge_pe_price - hedge_pe_entry)) * lot_size
                if total_pnl > max_pnl:
                    max_pnl = total_pnl

                if total_pnl >= target:
                    if not ce_exit_price:
                        ce_exit_price = ce_price
                    if not pe_exit_price:
                        pe_exit_price = pe_price
                    hedge_ce_exit_price = hedge_ce_exit_price or hedge_ce_price
                    hedge_pe_exit_price = hedge_pe_exit_price or hedge_pe_price
                    exit_ts = dt.datetime.now(IST)  # Always use IST
                    logging.info("Target hit. CE exit %.2f PE exit %.2f HedgeCE %.2f HedgePE %.2f total_pnl %.2f",
                                 ce_exit_price, pe_exit_price, hedge_ce_exit_price, hedge_pe_exit_price, total_pnl)
                    break

                if (ce_exit and pe_exit):
                    logging.info("Both legs exited by SL")
                    break

                time.sleep(60)
    else:
        logging.info("WS monitor disabled; using polling loop")
        while True:
            now = dt.datetime.now(IST).time()
            if now >= EXIT_TIME:
                live = _fetch_live_quotes(kite, api_key, enctoken, user_id,
                                          [ce_token, pe_token, hedge_ce_token, hedge_pe_token], symbols, last_prices)
                ce_exit_price = ce_exit_price or live.get(ce_token, {}).get('last_price', ce_entry)
                pe_exit_price = pe_exit_price or live.get(pe_token, {}).get('last_price', pe_entry)
                hedge_ce_exit_price = hedge_ce_exit_price or live.get(hedge_ce_token, {}).get('last_price',
                                                                                              hedge_ce_entry)
                hedge_pe_exit_price = hedge_pe_exit_price or live.get(hedge_pe_token, {}).get('last_price',
                                                                                              hedge_pe_entry)
                logging.info("Exit time reached. CE exit %.2f PE exit %.2f HedgeCE %.2f HedgePE %.2f", ce_exit_price,
                             pe_exit_price, hedge_ce_exit_price, hedge_pe_exit_price)
                exit_ts = dt.datetime.now(IST)
                break

            live = _fetch_live_quotes(kite, api_key, enctoken, user_id,
                                      [ce_token, pe_token, hedge_ce_token, hedge_pe_token], symbols, last_prices)
            ce_price = live.get(ce_token, {}).get('last_price', last_prices.get(ce_token, ce_entry))
            pe_price = live.get(pe_token, {}).get('last_price', last_prices.get(pe_token, pe_entry))
            hedge_ce_price = live.get(hedge_ce_token, {}).get('last_price',
                                                              last_prices.get(hedge_ce_token, hedge_ce_entry))
            hedge_pe_price = live.get(hedge_pe_token, {}).get('last_price',
                                                              last_prices.get(hedge_pe_token, hedge_pe_entry))
            last_prices[ce_token] = ce_price
            last_prices[pe_token] = pe_price
            last_prices[hedge_ce_token] = hedge_ce_price
            last_prices[hedge_pe_token] = hedge_pe_price
            logging.info(
                "Live CE %.2f PE %.2f | HedgeCE %.2f HedgePE %.2f | SLs CE %.2f PE %.2f | Target %.2f | LossAmt %.2f",
                ce_price, pe_price, hedge_ce_price, hedge_pe_price, ce_sl, pe_sl, target, loss_amount)

            if not ce_exit and ce_price >= ce_sl:
                ce_exit = 'SL'
                ce_exit_price = ce_price
                pe_sl = pe_price * (1 + STOP_LOSS_PCT)
                hedge_ce_exit_price = hedge_ce_price
                hedge_pe_exit_price = hedge_pe_exit_price
                exit_ts = dt.datetime.now(IST)
                logging.info("CE SL hit at %.2f. Closing hedges CE %.2f PE %.2f", ce_exit_price, hedge_ce_exit_price,
                             hedge_pe_exit_price)
            if not pe_exit and pe_price >= pe_sl:
                pe_exit = 'SL'
                pe_exit_price = pe_price
                ce_sl = ce_price * (1 + STOP_LOSS_PCT)
                hedge_ce_exit_price = hedge_ce_price
                hedge_pe_exit_price = hedge_pe_exit_price
                exit_ts = dt.datetime.now(IST)
                logging.info("PE SL hit at %.2f. Closing hedges CE %.2f PE %.2f", pe_exit_price, hedge_ce_exit_price,
                             hedge_pe_exit_price)

            total_pnl = ((ce_entry - (ce_exit_price or ce_price)) + (pe_entry - (pe_exit_price or pe_price)) + (
                        hedge_ce_price - hedge_ce_entry) + (hedge_pe_price - hedge_pe_entry)) * lot_size
            if total_pnl > max_pnl:
                max_pnl = total_pnl

            if total_pnl >= target:
                if not ce_exit_price:
                    ce_exit_price = ce_price
                if not pe_exit_price:
                    pe_exit_price = pe_price
                hedge_ce_exit_price = hedge_ce_exit_price or hedge_ce_price
                hedge_pe_exit_price = hedge_pe_exit_price or hedge_pe_price
                exit_ts = dt.datetime.now(IST)
                logging.info("Target hit. CE exit %.2f PE exit %.2f HedgeCE %.2f HedgePE %.2f total_pnl %.2f",
                             ce_exit_price, pe_exit_price, hedge_ce_exit_price, hedge_pe_exit_price, total_pnl)
                break

            if ce_exit and pe_exit:
                logging.info("Both legs exited by SL")
                break

            time.sleep(60)

    # Fallback exit_ts if still None
    if exit_ts is None:
        exit_ts = dt.datetime.now(IST)  # Always use IST

    profit = ((ce_entry - ce_exit_price) + (pe_entry - pe_exit_price) + (hedge_ce_exit_price - hedge_ce_entry) + (
                hedge_pe_exit_price - hedge_pe_entry)) * lot_size
    log = [{
        'date': dt.datetime.now(IST).date(),  # Always use IST
        'entry_time': entry_ts,
        'exit_time': exit_ts,
        'strike': int(atm_strike),
        'call_entry': ce_entry,
        'put_entry': pe_entry,
        'call_exit': ce_exit_price,
        'put_exit': pe_exit_price,
        'hedge_ce_entry': hedge_ce_entry,
        'hedge_pe_entry': hedge_pe_entry,
        'hedge_ce_exit': hedge_ce_exit_price,
        'hedge_pe_exit': hedge_pe_exit_price,
        'lot_size': lot_size,
        'call_sl': ce_sl,
        'put_sl': pe_sl,
        'target_amount': target,
        'loss_amount': loss_amount,
        'profit': profit,
        'max_intraday_pnl': max_pnl,
        'call_symbol': ce_symbol,
        'put_symbol': pe_symbol,
        'hedge_call_symbol': hedge_ce_symbol,
        'hedge_put_symbol': hedge_pe_symbol,
    }]
    df_log = pd.DataFrame(log)
    header = not pd.io.common.file_exists(LOG_FILE)
    df_log.to_csv(LOG_FILE, mode='a', header=header, index=False)
    logging.info("Logged trade %s", log[0])
    _send_trade_summary_email(cfg, log[0])
    print(f"Logged trade: {log[0]}")


if __name__ == '__main__':
    logging.info("Starting papertrade")
    # # Test direct order placement and print response
    # try:
    #     from auth_utils import get_kite_with_refresh  # FIX: import from auth_utils, not kite_trade
    #     cfg = load_config()
    #     kite, _ = get_kite_with_refresh(cfg)
    #     order = kite.place_order(
    #         variety=kite.VARIETY_REGULAR,
    #         exchange=kite.EXCHANGE_NSE,
    #         tradingsymbol="ACC",
    #         transaction_type=kite.TRANSACTION_TYPE_BUY,
    #         quantity=1,
    #         product=kite.PRODUCT_MIS,
    #         order_type=kite.ORDER_TYPE_MARKET,
    #         price=None,
    #         validity=None,
    #         disclosed_quantity=None,
    #         trigger_price=None,
    #         squareoff=None,
    #         stoploss=None,
    #         trailing_stoploss=None,
    #         tag="AlgoTrader"
    #     )
    #     print("Direct order response:", order)
    #     logging.info(f"Direct order response: {order}")
    # except Exception as e:
    #     import traceback
    #     tb = traceback.format_exc()
    #     print(f"Direct order failed: {e}\n{tb}")
    #     logging.error(f"Direct order failed: {e}\n{tb}")
    paper_trade()

