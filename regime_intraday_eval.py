import argparse
import pathlib
import pandas as pd


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
    return df


def classify_intraday_day(day_df: pd.DataFrame, range_pct_th: float, move_pct_th: float) -> str:
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


def evaluate(intraday_path: str, regime_path: str, out_csv: str, range_pct_th: float, move_pct_th: float):
    intraday = load_df(intraday_path)
    regime = load_df(regime_path)
    if "trend" not in regime.columns:
        raise ValueError("Regime CSV must have a 'trend' column from regime_backtest")

    intraday["d"] = intraday["date"].dt.date
    regime["d"] = pd.to_datetime(regime["date"]).dt.date

    records = []
    for d, day_df in intraday.groupby("d"):
        realized = classify_intraday_day(day_df.sort_values("date"), range_pct_th, move_pct_th)
        pred_row = regime[regime["d"] == d]
        if pred_row.empty:
            continue
        pred = pred_row.iloc[0]["trend"]
        o = day_df.iloc[0]["open"]
        h = day_df["high"].max()
        l = day_df["low"].min()
        c = day_df.iloc[-1]["close"]
        range_pct = (h - l) / o * 100 if o else 0
        move_pct = (c - o) / o * 100 if o else 0
        records.append({
            "date": d,
            "predicted_trend": pred,
            "realized_label": realized,
            "range_pct": round(range_pct, 3),
            "move_pct": round(move_pct, 3),
            "match": pred == realized
        })

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    if not df.empty:
        match_rate = df["match"].mean() * 100
        realized_counts = df["realized_label"].value_counts().to_dict()
        pred_counts = df["predicted_trend"].value_counts().to_dict()
        print(f"Saved {len(df)} rows to {out_csv}")
        print(f"Match rate: {match_rate:.2f}%")
        print(f"Pred counts: {pred_counts}")
        print(f"Realized counts: {realized_counts}")
    else:
        print("No overlapping days between intraday and regime data")


def main():
    ap = argparse.ArgumentParser(description="Compare daily trend regime vs realized intraday behavior")
    ap.add_argument("--intraday_path", required=True, help="Path to intraday candles (5m) for index")
    ap.add_argument("--regime_csv", required=True, help="Path to regime_backtest output CSV (with 'trend' column)")
    ap.add_argument("--out_csv", default="regime_intraday_eval.csv", help="Where to write comparison results")
    ap.add_argument("--range_pct_th", type=float, default=0.8, help="Max intraday range%% to call SIDEWAYS")
    ap.add_argument("--move_pct_th", type=float, default=0.3, help="Max close-open move%% to call SIDEWAYS")
    args = ap.parse_args()

    evaluate(args.intraday_path, args.regime_csv, args.out_csv, args.range_pct_th, args.move_pct_th)


if __name__ == "__main__":
    main()

