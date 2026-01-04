import os
import sys
import argparse
import datetime as dt
import pandas as pd
from typing import List

from auth_utils import load_config, get_kite_with_refresh
from instrument_cache import InstrumentCache


def chunked_historical(kite, token: int, start: dt.datetime, end: dt.datetime, interval: str, oi: bool = False):
    """Fetch historical data in 100-day chunks to avoid API limits."""
    curr = start
    out = []
    while curr < end:
        chunk_end = min(curr + dt.timedelta(days=100), end)
        try:
            data = kite.historical_data(token, curr, chunk_end, interval, continuous=False, oi=oi)
            if data:
                out.extend(data)
            print(f"Fetched {len(data) if data else 0} rows for token {token} from {curr.date()} to {chunk_end.date()}")
        except Exception as e:
            print(f"Error fetching {curr.date()}->{chunk_end.date()} for token {token}: {e}")
        curr = chunk_end
    return out


def save_df(df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    if out_path.endswith('.parquet'):
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


def collect_index(kite, ic: InstrumentCache, underlying: str, lookback_days: int, interval: str, out_dir: str):
    token = ic.get_index_token(underlying)
    end = dt.datetime.now()
    start = end - dt.timedelta(days=lookback_days)
    data = chunked_historical(kite, token, start, end, interval, oi=False)
    df = pd.DataFrame(data)
    if not df.empty:
        df.drop_duplicates(subset=['date'], inplace=True)
        df.sort_values('date', inplace=True)
    save_df(df, os.path.join(out_dir, f"{underlying}_index_{interval}.parquet"))


def collect_index_range(kite, token: int, start: dt.datetime, end: dt.datetime, interval: str, out_path: str):
    data = chunked_historical(kite, token, start, end, interval, oi=False)
    df = pd.DataFrame(data)
    if not df.empty:
        df.drop_duplicates(subset=['date'], inplace=True)
        df.sort_values('date', inplace=True)
    save_df(df, out_path)


def collect_options(kite, ic: InstrumentCache, underlying: str, expiry: dt.date, strikes: List[float], interval: str, lookback_days: int, out_dir: str):
    end = dt.datetime.now()
    start = end - dt.timedelta(days=lookback_days)
    for opt_type in ['CE', 'PE']:
        for strike in strikes:
            inst = ic.find_option(underlying, expiry, strike, opt_type)
            if not inst:
                print(f"Skip {underlying} {expiry} {strike} {opt_type}: not found")
                continue
            data = chunked_historical(kite, inst['instrument_token'], start, end, interval, oi=True)
            df = pd.DataFrame(data)
            if not df.empty:
                df.drop_duplicates(subset=['date'], inplace=True)
                df.sort_values('date', inplace=True)
            fname = f"{underlying}_{expiry}_{int(strike)}{opt_type}_{interval}.parquet"
            save_df(df, os.path.join(out_dir, fname))


def main():
    parser = argparse.ArgumentParser(description="Collect historical data for index and options")
    parser.add_argument('--underlying', default='BANKNIFTY', choices=['BANKNIFTY', 'NIFTY'])
    parser.add_argument('--interval', default='5minute')
    parser.add_argument('--lookback_days', type=int, default=365)
    parser.add_argument('--out_dir', default='data')
    parser.add_argument('--weekly', action='store_true', help='Use nearest weekly expiry if available')
    parser.add_argument('--wing_pct', type=float, default=1.0, help='Distance from spot for option strikes (percent)')
    parser.add_argument('--skip_options', action='store_true', help='Skip option collection (index only)')
    parser.add_argument('--daily_start', default='1990-01-01', help='Start date for full-history daily candles (YYYY-MM-DD)')
    parser.add_argument('--intraday_lookback_days', type=int, default=1825, help='Lookback days for intraday interval (e.g., 5y=1825)')
    args = parser.parse_args()

    cfg = load_config()
    kite, enctoken = get_kite_with_refresh(cfg)

    ic = InstrumentCache.from_files_or_api(kite)

    # Index daily from inception
    token = ic.get_index_token(args.underlying)
    daily_start = dt.datetime.fromisoformat(args.daily_start)
    daily_out = os.path.join(args.out_dir, f"{args.underlying}_index_day_full.parquet")
    collect_index_range(kite, token, daily_start, dt.datetime.now(), 'day', daily_out)

    # Index intraday (e.g., 5y of 5m)
    end = dt.datetime.now()
    start_intraday = end - dt.timedelta(days=args.intraday_lookback_days)
    intraday_out = os.path.join(args.out_dir, f"{args.underlying}_index_{args.interval}_{args.intraday_lookback_days}d.parquet")
    collect_index_range(kite, token, start_intraday, end, args.interval, intraday_out)

    if args.skip_options:
        return

    expiry = ic.get_nearest_expiry(args.underlying, weekly_first=args.weekly)
    spot = ic.get_spot_from_ws(kite, args.underlying, cfg.get('auth', {}).get('api_key'), enctoken, kite.profile().get('user_id'))
    ce_strike = ic.round_to_strike(spot * (1 + args.wing_pct / 100), args.underlying)
    pe_strike = ic.round_to_strike(spot * (1 - args.wing_pct / 100), args.underlying)
    strikes = sorted({ce_strike, pe_strike})

    collect_options(kite, ic, args.underlying, expiry, strikes, args.interval, args.lookback_days, args.out_dir)


if __name__ == '__main__':
    sys.exit(main())

