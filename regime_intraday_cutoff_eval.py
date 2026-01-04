import argparse
import datetime as dt
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
    return df.sort_values("date").reset_index(drop=True)


def classify_slice(df: pd.DataFrame, range_pct_th: float, move_pct_th: float) -> str:
    if df.empty:
        return SIDEWAYS
    o = df.iloc[0]["open"]
    h = df["high"].max()
    l = df["low"].min()
    c = df.iloc[-1]["close"]
    range_pct = (h - l) / o * 100 if o else 0
    move_pct = (c - o) / o * 100 if o else 0
    if range_pct <= range_pct_th and abs(move_pct) <= move_pct_th:
        return SIDEWAYS
    if move_pct > 0:
        return UP
    return DOWN


def evaluate(intraday_path: str, cutoff: str, out_csv: str, range_pct_th: float, move_pct_th: float):
    intraday = load_df(intraday_path)
    intraday["d"] = intraday["date"].dt.date
    cutoff_time = dt.datetime.strptime(cutoff, "%H:%M").time()

    rows = []
    for d, day_df in intraday.groupby("d"):
        day_df = day_df.sort_values("date")
        day_df["t"] = day_df["date"].dt.time
        pre = day_df[day_df["t"] <= cutoff_time]
        post = day_df[day_df["t"] > cutoff_time]
        if pre.empty or post.empty:
            continue
        pred = classify_slice(pre, range_pct_th, move_pct_th)
        realized = classify_slice(post, range_pct_th, move_pct_th)
        rows.append({
            "date": d,
            "predicted_label": pred,
            "realized_label": realized,
            "match": pred == realized
        })
    res = pd.DataFrame(rows)
    res.to_csv(out_csv, index=False)
    if res.empty:
        print("No days evaluated; check data/cutoff")
        return
    match_rate = res["match"].mean() * 100
    pred_counts = res["predicted_label"].value_counts().to_dict()
    realized_counts = res["realized_label"].value_counts().to_dict()
    print(f"Saved {len(res)} rows to {out_csv}")
    print(f"Match rate: {match_rate:.2f}%")
    print(f"Pred counts: {pred_counts}")
    print(f"Realized counts: {realized_counts}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate intraday regime snapshot at cutoff vs rest-of-day")
    ap.add_argument("--intraday_path", required=True, help="Path to intraday candles (5m)")
    ap.add_argument("--cutoff", default="10:00", help="Cutoff time HH:MM for snapshot (e.g., 10:00)")
    ap.add_argument("--out_csv", default="regime_intraday_cutoff_eval.csv", help="Where to write results")
    ap.add_argument("--range_pct_th", type=float, default=0.8, help="Max range%% to call SIDEWAYS")
    ap.add_argument("--move_pct_th", type=float, default=0.3, help="Max move%% to call SIDEWAYS")
    args = ap.parse_args()

    evaluate(args.intraday_path, args.cutoff, args.out_csv, args.range_pct_th, args.move_pct_th)


if __name__ == "__main__":
    main()

