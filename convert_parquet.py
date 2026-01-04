import pathlib
import pandas as pd

DATA_DIR = pathlib.Path("data")
OUT_DIR = DATA_DIR / "csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for p in DATA_DIR.glob("*.parquet"):
    df = pd.read_parquet(p)
    out = OUT_DIR / (p.stem + ".csv")
    df.to_csv(out, index=False)
    print(f"Converted {p.name} -> {out}")

