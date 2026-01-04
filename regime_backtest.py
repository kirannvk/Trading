import argparse
import datetime as dt
import pathlib
import pandas as pd
from regime import classify_trend_daily, classify_vol_daily, MarketRegime


def load_daily(path: str) -> pd.DataFrame:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Daily file not found: {path}")
    if p.suffix.lower() == '.parquet':
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    if 'date' not in df.columns:
        raise ValueError("Daily file must have a 'date' column")
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)


def backtest_regime(daily_df: pd.DataFrame, market_start=dt.time(9, 15), market_end=dt.time(15, 30)) -> pd.DataFrame:
    results = []
    for i in range(len(daily_df)):
        window = daily_df.iloc[: i + 1]
        try:
            trend = classify_trend_daily(window)
            vol = classify_vol_daily(window)
        except ValueError:
            continue
        tradable = True
        reason = "OK"
        if vol.volatility == "HIGH":
            tradable = False
            reason = "High volatility"
        results.append({
            'date': window.iloc[-1]['date'].date(),
            'trend': trend.trend,
            'trend_confidence': trend.confidence,
            'volatility': vol.volatility,
            'vol_percentile': vol.percentile,
            'vol_confidence': vol.confidence,
            'tradable': tradable,
            'reason': reason,
        })
    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser(description="Backtest daily regime (trend/vol) on historical daily data")
    ap.add_argument('--daily_path', required=True, help='Path to daily candles (parquet/csv)')
    ap.add_argument('--out_csv', default='regime_backtest_output.csv', help='Where to write regime timeline')
    args = ap.parse_args()

    daily_df = load_daily(args.daily_path)
    res = backtest_regime(daily_df)
    res.to_csv(args.out_csv, index=False)
    trend_counts = res['trend'].value_counts().to_dict()
    vol_counts = res['volatility'].value_counts().to_dict()
    tradable_pct = res['tradable'].mean() * 100 if not res.empty else 0
    print(f"Saved {len(res)} rows to {args.out_csv}")
    print(f"Trend counts: {trend_counts}")
    print(f"Vol counts: {vol_counts}")
    print(f"Tradable %: {tradable_pct:.1f}")


if __name__ == '__main__':
    main()

