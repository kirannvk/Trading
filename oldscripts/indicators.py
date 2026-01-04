# market_intel/indicators.py

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def true_range(high, low, prev_close):
    return np.maximum(
        high - low,
        np.maximum(
            abs(high - prev_close),
            abs(low - prev_close)
        )
    )


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr = true_range(high, low, prev_close)

    return tr.rolling(window=period).mean()


def percentile_rank(series: pd.Series, value: float) -> float:
    """
    Returns percentile rank (0–100) of value within series
    """
    return (series < value).mean() * 100
