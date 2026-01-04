import argparse
import os
import pandas as pd
import datetime as dt
import threading
import time
from typing import List, Dict
from kiteconnect import KiteTicker
import logging

from auth_utils import load_config, get_kite_with_refresh
from instrument_cache import InstrumentCache, INDEX_TOKENS

# Silence noisy WS close logs from kiteconnect
logging.getLogger("kiteconnect.ticker").setLevel(logging.WARNING)
logging.getLogger("websocket").setLevel(logging.WARNING)


def _cleanup_kws(kws, tokens=None):
    """Best-effort unsubscribe and close for KiteTicker."""
    try:
        if tokens:
            kws.unsubscribe(tokens)
    except Exception:
        pass
    try:
        kws.close()
    except Exception:
        pass


def fetch_spot_ws(api_key: str, enctoken: str, user_id: str, token: int, timeout: int = 8) -> float:
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
        ws.set_mode(ws.MODE_FULL, [token])

    def on_error(ws, code, reason):
        logging.warning("fetch_spot_ws error code=%s reason=%s", code, reason)

    def on_close(ws, code, reason):
        logging.warning("fetch_spot_ws closed code=%s reason=%s", code, reason)

    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    kws.on_error = on_error
    kws.on_close = on_close
    threading.Thread(target=kws.connect, kwargs={'threaded': True}, daemon=True).start()
    done.wait(timeout)
    time.sleep(0.5)
    _cleanup_kws(kws, [token])
    return result['ltp']


def fetch_option_quotes_ws(api_key: str, enctoken: str, user_id: str, tokens: List[int], timeout: int = 12) -> Dict[int, dict]:
    result: Dict[int, dict] = {}
    done = threading.Event()
    remaining = set(tokens)
    kws = KiteTicker(api_key=api_key, access_token=f"{enctoken}&user_id={user_id}")

    def on_ticks(ws, ticks):
        nonlocal remaining
        for t in ticks:
            tok = t.get('instrument_token')
            if tok in remaining:
                result[tok] = t
                remaining.discard(tok)
        if not remaining:
            done.set()
            ws.close()

    def on_connect(ws, response):
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_FULL, tokens)

    def on_error(ws, code, reason):
        logging.warning("fetch_option_quotes_ws error code=%s reason=%s", code, reason)

    def on_close(ws, code, reason):
        logging.warning("fetch_option_quotes_ws closed code=%s reason=%s", code, reason)

    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    kws.on_error = on_error
    kws.on_close = on_close
    threading.Thread(target=kws.connect, kwargs={'threaded': True}, daemon=True).start()
    done.wait(timeout)
    time.sleep(0.5)
    _cleanup_kws(kws, tokens)
    return result


def fetch_option_chain(kite, ic: InstrumentCache, underlying: str, strikes_around_atm: int = 5, expiry: dt.date = None, spot: float = None,
                       api_key: str = None, enctoken: str = None, user_id: str = None):
    # Use provided spot; fallback only to nearest future last_price if missing
    if not spot or spot <= 0:
        futs = ic.nfo[(ic.nfo['instrument_type'] == 'FUT') & (ic.nfo['tradingsymbol'].str.startswith(underlying))]
        if not futs.empty:
            fut = futs.sort_values('expiry').iloc[0]
            spot = fut.get('last_price', 0) or 0
    if not spot or spot <= 0:
        raise RuntimeError("Could not fetch spot (WS or futures fallback)")

    if expiry is None:
        expiry = ic.get_nearest_expiry(underlying, weekly_first=True)

    step = ic.round_to_strike(100, underlying) - ic.round_to_strike(0, underlying)  # use step mapping
    atm = ic.round_to_strike(spot, underlying)
    strikes = [atm + step * i for i in range(-strikes_around_atm, strikes_around_atm + 1)]

    rows = []
    symbols = []
    tokens = []
    for opt_type in ["CE", "PE"]:
        for strike in strikes:
            inst = ic.find_option(underlying, expiry, strike, opt_type)
            if not inst:
                continue
            symbols.append(f"{inst['exchange']}:{inst['tradingsymbol']}")
            tokens.append(int(inst['instrument_token']))
            rows.append({
                "symbol": inst['tradingsymbol'],
                "token": inst['instrument_token'],
                "expiry": expiry,
                "strike": strike,
                "type": opt_type,
                "lot_size": inst.get('lot_size', None),
            })

    if not tokens:
        raise RuntimeError("No options found for the selected range")

    quote_data = {}
    if api_key and enctoken and user_id:
        quote_data = fetch_option_quotes_ws(api_key=api_key, enctoken=enctoken, user_id=user_id, tokens=tokens)
    if not quote_data:
        quote = fetch_quotes(kite, symbols)
        quote_data = {}
        for r in rows:
            key = f"NFO:{r['symbol']}"
            q = quote.get(key) or quote.get(r['symbol']) or {}
            quote_data[int(r['token'])] = q

    for r in rows:
        q = quote_data.get(int(r['token']), {})
        depth = q.get('depth', {}) if isinstance(q, dict) else {}
        if depth:
            buy_depth = depth.get('buy', [])
            sell_depth = depth.get('sell', [])
            bid = buy_depth[0].get('price') if buy_depth else None
            ask = sell_depth[0].get('price') if sell_depth else None
        else:
            bid = None
            ask = None
        # volume in WS tick is volume_traded
        vol_field = q.get('volume') if isinstance(q, dict) else None
        if vol_field is None and isinstance(q, dict):
            vol_field = q.get('volume_traded')
        r.update({
            "ltp": q.get("last_price") if isinstance(q, dict) else q.get('last_price', None) if hasattr(q, 'get') else None,
            "volume": vol_field,
            "oi": q.get("oi") if isinstance(q, dict) else None,
            "bid": bid,
            "ask": ask,
        })
    df = pd.DataFrame(rows).sort_values(["strike", "type"])
    return spot, expiry, df


def fetch_quotes(kite, symbols):
    try:
        resp = kite.session.get(f"{kite.root_url}/quote", params={"i": symbols}, headers=kite.headers).json()
        return resp.get("data", {}) if isinstance(resp, dict) else {}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser(description="Fetch option chain snapshot around ATM")
    ap.add_argument("--underlying", choices=["BANKNIFTY", "NIFTY"], default="BANKNIFTY")
    ap.add_argument("--strikes", type=int, default=5, help="Number of strikes on each side of ATM")
    ap.add_argument("--out_csv", default="option_chain.csv")
    args = ap.parse_args()

    cfg = load_config()
    kite, enctoken = get_kite_with_refresh(cfg)
    ic = InstrumentCache.from_files_or_api(kite)

    underlying = cfg.get('strategy', {}).get('underlying', args.underlying)
    api_key = cfg.get('auth', {}).get('api_key')
    profile = kite.profile()
    user_id = profile.get('user_id') if profile else None

    # Spot via WebSocket token; fallback to future last_price
    tok = ic.get_index_token(underlying)
    spot = 0
    if api_key and enctoken and user_id:
        spot = fetch_spot_ws(api_key, enctoken, user_id, tok, timeout=5)
    if not spot:
        futs = ic.nfo[(ic.nfo['instrument_type'] == 'FUT') & (ic.nfo['tradingsymbol'].str.startswith(underlying))]
        if not futs.empty:
            fut = futs.sort_values('expiry').iloc[0]
            spot = fut.get('last_price', 0) or 0
    if not spot:
        raise RuntimeError("Could not fetch spot via WS or futures fallback")

    spot, expiry, df = fetch_option_chain(kite, ic, underlying, strikes_around_atm=args.strikes, spot=spot,
                                          api_key=api_key, enctoken=enctoken, user_id=user_id)
    df.to_csv(args.out_csv, index=False)
    print(f"Underlying {underlying} spot ~ {spot:.2f}, expiry {expiry}")
    print(f"Saved {len(df)} rows to {args.out_csv}")
    print(df.head(10))


if __name__ == "__main__":
    main()

