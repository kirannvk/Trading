# market_intel/regime_logic.py

from dataclasses import dataclass
import pandas as pd
from market_intel.indicators import ema, atr, percentile_rank
from config.settings import EMA_FAST, EMA_SLOW, ATR_PERIOD, ATR_LOOKBACK_DAYS


@dataclass
class TrendRegime:
    trend: str            # UP / DOWN / SIDEWAYS
    ema_fast: float
    ema_slow: float
    confidence: float


def classify_trend(daily_df: pd.DataFrame) -> TrendRegime:
    """
    Classify market trend using daily EMA50 / EMA200
    daily_df must be sorted by timestamp ascending
    """

    if len(daily_df) < EMA_SLOW + 5:
        raise ValueError("Not enough data to compute trend regime")

    closes = daily_df["close"]

    ema_fast_series = ema(closes, EMA_FAST)
    ema_slow_series = ema(closes, EMA_SLOW)

    ema_fast_val = ema_fast_series.iloc[-1]
    ema_slow_val = ema_slow_series.iloc[-1]
    price = closes.iloc[-1]

    # Conservative buffer: 0.2% of price
    buffer = price * 0.002

    if ema_fast_val > ema_slow_val + buffer:
        trend = "UP"
    elif ema_fast_val < ema_slow_val - buffer:
        trend = "DOWN"
    else:
        trend = "SIDEWAYS"

    # Confidence based on EMA separation
    separation = abs(ema_fast_val - ema_slow_val)
    confidence = min(1.0, separation / (price * 0.01))

    return TrendRegime(
        trend=trend,
        ema_fast=ema_fast_val,
        ema_slow=ema_slow_val,
        confidence=round(confidence, 2)
    )



@dataclass
class VolatilityRegime:
    volatility: str        # LOW / NORMAL / HIGH
    atr_value: float
    percentile: float
    confidence: float


def classify_volatility(daily_df: pd.DataFrame) -> VolatilityRegime:
    """
    Classify market volatility using ATR percentile
    daily_df must be sorted by timestamp ascending
    """

    if len(daily_df) < ATR_LOOKBACK_DAYS + ATR_PERIOD:
        raise ValueError("Not enough data to compute volatility regime")

    # Compute ATR
    daily_df = daily_df.copy()
    daily_df["atr"] = atr(daily_df, ATR_PERIOD)

    atr_series = daily_df["atr"].dropna()

    current_atr = atr_series.iloc[-1]
    atr_history = atr_series.iloc[-ATR_LOOKBACK_DAYS:]

    pctl = percentile_rank(atr_history, current_atr)

    # Volatility classification
    if pctl < 30:
        volatility = "LOW"
    elif pctl > 70:
        volatility = "HIGH"
    else:
        volatility = "NORMAL"

    # Confidence: distance from regime boundaries
    if volatility == "LOW":
        confidence = (30 - pctl) / 30
    elif volatility == "HIGH":
        confidence = (pctl - 70) / 30
    else:
        confidence = 1 - abs(pctl - 50) / 20

    confidence = round(max(0.0, min(confidence, 1.0)), 2)

    return VolatilityRegime(
        volatility=volatility,
        atr_value=round(current_atr, 2),
        percentile=round(pctl, 2),
        confidence=confidence
    )
