import argparse
import pathlib
import datetime as dt
import pandas as pd

from regime import classify_trend_daily, classify_vol_daily

SIDEWAYS = "SIDEWAYS"
UP = "UP"
DOWN = "DOWN"


def load_df(path: str) -> pd.DataFrame:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    if "date" not in df.columns:
        raise ValueError("Data must have a 'date' column")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def classify_realized_intraday(day_df: pd.DataFrame, range_pct_th: float, move_pct_th: float) -> str:
    if day_df.empty:
        return SIDEWAYS
    o = day_df.iloc[0]["open"]
    h = day_df["high"].max()
    l = day_df["low"].min()
    c = day_df.iloc[-1]["close"]
    range_pct = (h - l) / o * 100 if o else 0
    move_pct = (c - o) / o * 100 if o else 0
    if range_pct <= range_pct_th and abs(move_pct) <= move_pct_th:
        return SIDEWAYS
    if move_pct > 0:
        return UP
    return DOWN


def evaluate(daily_path: str, intraday_path: str, out_csv: str, range_pct_th: float, move_pct_th: float):
    daily = load_df(daily_path)
    intraday = load_df(intraday_path)
    intraday["d"] = intraday["date"].dt.date

    records = []
    for i in range(1, len(daily)):
        today = daily.iloc[i]
        as_of_date = today["date"].date()
        # use data up to previous day to generate a prior-day regime prediction
        window = daily.iloc[:i]
        try:
            trend = classify_trend_daily(window)
            vol = classify_vol_daily(window)
        except ValueError:
            continue
        # realized intraday for this day
        day_df = intraday[intraday["d"] == as_of_date].sort_values("date")
        realized = classify_realized_intraday(day_df, range_pct_th, move_pct_th)
        records.append({
            "date": as_of_date,
            "predicted_trend": trend.trend,
            "pred_trend_confidence": trend.confidence,
            "pred_volatility": vol.volatility,
            "realized_label": realized,
            "match": trend.trend == realized
        })

    res = pd.DataFrame(records)
    res.to_csv(out_csv, index=False)
    if res.empty:
        print("No overlapping days to evaluate")
        return
    match_rate = res["match"].mean() * 100
    pred_counts = res["predicted_trend"].value_counts().to_dict()
    realized_counts = res["realized_label"].value_counts().to_dict()
    print(f"Saved {len(res)} rows to {out_csv}")
    print(f"Match rate: {match_rate:.2f}%")
    print(f"Pred counts: {pred_counts}")
    print(f"Realized counts: {realized_counts}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate prior-day regime vs next-day intraday behavior")
    ap.add_argument("--daily_path", required=True, help="Path to daily candles (parquet/csv)")
    ap.add_argument("--intraday_path", required=True, help="Path to intraday candles (5m) for same index")
    ap.add_argument("--out_csv", default="regime_intraday_prediction_eval.csv", help="Where to write comparison results")
    ap.add_argument("--range_pct_th", type=float, default=0.8, help="Max intraday range%% to call SIDEWAYS")
    ap.add_argument("--move_pct_th", type=float, default=0.3, help="Max close-open move%% to call SIDEWAYS")
    args = ap.parse_args()

    evaluate(args.daily_path, args.intraday_path, args.out_csv, args.range_pct_th, args.move_pct_th)


if __name__ == "__main__":
    main()

