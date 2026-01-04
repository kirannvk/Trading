import datetime as dt
import pandas as pd
from typing import Optional, Dict
import logging
from dataclasses import dataclass

from indicators import ema, atr, percentile_rank

# Simple rule-based regime classifier
TREND = "TREND"
RANGE = "RANGE"
EVENT = "EVENT"

# Defaults for daily regime
EMA_FAST = 50
EMA_SLOW = 200
ATR_PERIOD = 14
ATR_LOOKBACK_DAYS = 200
VOL_LOW_PCTL = 30
VOL_HIGH_PCTL = 70

logger = logging.getLogger(__name__)


@dataclass
class TrendRegime:
    trend: str  # UP / DOWN / SIDEWAYS
    ema_fast: float
    ema_slow: float
    confidence: float


@dataclass
class VolatilityRegime:
    volatility: str  # LOW / NORMAL / HIGH
    atr_value: float
    percentile: float
    confidence: float


@dataclass
class MarketRegime:
    trend: TrendRegime
    volatility: VolatilityRegime
    tradable: bool
    reason: str


def _opening_range(day_df: pd.DataFrame, minutes: int = 30) -> pd.DataFrame:
    start_time = day_df['date'].min()
    cutoff = start_time + pd.Timedelta(minutes=minutes)
    return day_df[day_df['date'] < cutoff]


def classify_day(day_df: pd.DataFrame, prev_close: float, vix_change: float = 0.0,
                 opening_range_minutes: int = 30, gap_threshold: float = 0.8) -> str:
    """Classify a single day's regime using gap, opening range, and VIX change.

    - EVENT: large gap or big VIX spike
    - TREND: opening range breakout with flat/up VIX
    - RANGE: otherwise
    """
    if day_df.empty:
        logger.debug("Regime: empty day_df -> RANGE")
        return RANGE
    day_df = day_df.sort_values('date')
    first_open = day_df.iloc[0]['open']
    gap_pct = ((first_open - prev_close) / prev_close) * 100 if prev_close else 0

    if abs(gap_pct) >= gap_threshold or vix_change >= 5:
        logger.debug("Regime: EVENT due to gap_pct=%.2f vix_change=%.2f", gap_pct, vix_change)
        return EVENT

    orb = _opening_range(day_df, minutes=opening_range_minutes)
    if orb.empty:
        return RANGE
    orb_high = orb['high'].max()
    orb_low = orb['low'].min()
    day_high = day_df['high'].max()
    day_low = day_df['low'].min()

    breakout_up = day_high > orb_high * 1.001  # tiny buffer
    breakout_down = day_low < orb_low * 0.999

    if (breakout_up or breakout_down) and vix_change >= 0:
        regime = TREND
    else:
        regime = RANGE
    logger.debug("Regime: %s gap_pct=%.2f vix_change=%.2f orb_high=%.2f orb_low=%.2f", regime, gap_pct, vix_change, orb_high, orb_low)
    return regime


def classify_series(df: pd.DataFrame, vix_df: Optional[pd.DataFrame] = None,
                    opening_range_minutes: int = 30, gap_threshold: float = 0.8) -> Dict[dt.date, str]:
    """Return a map of date -> regime."""
    regimes = {}
    df = df.copy()
    df['d'] = df['date'].dt.date
    if vix_df is not None:
        vix_df = vix_df.copy()
        vix_df['d'] = vix_df['date'].dt.date
        vix_change_map = vix_df.groupby('d')['close'].apply(lambda s: (s.iloc[-1] - s.iloc[0]) / s.iloc[0] * 100)
    else:
        vix_change_map = {}

    grouped = df.groupby('d')
    prev_close = None
    for d, day in grouped:
        vix_chg = vix_change_map.get(d, 0.0)
        regime = classify_day(day, prev_close or day.iloc[0]['open'], vix_chg,
                               opening_range_minutes=opening_range_minutes,
                               gap_threshold=gap_threshold)
        logger.debug("Regime classify_series date=%s regime=%s vix_change=%.2f", d, regime, vix_chg)
        prev_close = day.iloc[-1]['close']
        regimes[d] = regime
    logger.info("Regime counts: %s", {r: list(regimes.values()).count(r) for r in set(regimes.values())})
    return regimes


def classify_trend_daily(daily_df: pd.DataFrame) -> TrendRegime:
    if len(daily_df) < EMA_SLOW + 5:
        raise ValueError("Not enough data to compute trend regime")
    closes = daily_df["close"]
    ema_fast_series = ema(closes, EMA_FAST)
    ema_slow_series = ema(closes, EMA_SLOW)
    ema_fast_val = ema_fast_series.iloc[-1]
    ema_slow_val = ema_slow_series.iloc[-1]
    price = closes.iloc[-1]
    buffer = price * 0.002
    if ema_fast_val > ema_slow_val + buffer:
        trend = "UP"
    elif ema_fast_val < ema_slow_val - buffer:
        trend = "DOWN"
    else:
        trend = "SIDEWAYS"
    separation = abs(ema_fast_val - ema_slow_val)
    confidence = min(1.0, separation / (price * 0.01))
    return TrendRegime(trend=trend, ema_fast=ema_fast_val, ema_slow=ema_slow_val, confidence=round(confidence, 2))


def classify_vol_daily(daily_df: pd.DataFrame) -> VolatilityRegime:
    if len(daily_df) < ATR_LOOKBACK_DAYS + ATR_PERIOD:
        raise ValueError("Not enough data to compute volatility regime")
    daily_df = daily_df.copy()
    daily_df["atr"] = atr(daily_df, ATR_PERIOD)
    atr_series = daily_df["atr"].dropna()
    current_atr = atr_series.iloc[-1]
    atr_history = atr_series.iloc[-ATR_LOOKBACK_DAYS:]
    pctl = percentile_rank(atr_history, current_atr)
    if pctl < VOL_LOW_PCTL:
        volatility = "LOW"
    elif pctl > VOL_HIGH_PCTL:
        volatility = "HIGH"
    else:
        volatility = "NORMAL"
    if volatility == "LOW":
        confidence = (VOL_LOW_PCTL - pctl) / VOL_LOW_PCTL
    elif volatility == "HIGH":
        confidence = (pctl - VOL_HIGH_PCTL) / (100 - VOL_HIGH_PCTL)
    else:
        confidence = 1 - abs(pctl - 50) / 20
    confidence = round(max(0.0, min(confidence, 1.0)), 2)
    return VolatilityRegime(volatility=volatility, atr_value=round(current_atr, 2), percentile=round(pctl, 2), confidence=confidence)


def classify_market_regime(daily_df: pd.DataFrame, as_of: Optional[dt.datetime] = None, market_start: dt.time = dt.time(9, 15), market_end: dt.time = dt.time(15, 30)) -> MarketRegime:
    as_of = as_of or dt.datetime.now()
    daily_df = daily_df.sort_values('date')
    trend_regime = classify_trend_daily(daily_df)
    vol_regime = classify_vol_daily(daily_df)
    tradable = True
    reason = "OK"
    t = as_of.time()
    if not (market_start <= t <= market_end):
        tradable = False
        reason = "Outside market hours"
    if vol_regime.volatility == "HIGH":
        tradable = False
        reason = "High volatility"
    return MarketRegime(trend=trend_regime, volatility=vol_regime, tradable=tradable, reason=reason)
