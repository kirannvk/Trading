from dataclasses import dataclass
import numpy as np
import pandas as pd

from db.candle_repository import CandleRepository


ATR_PERIOD = 14
ATR_LOOKBACK = 200          # last 200 x 5-min bars
ATR_SPIKE_PERCENTILE = 85


@dataclass
class IntradayOverride:
    active: bool
    reason: str
    atr_value: float
    percentile: float


class IntradayVolatilityOverrideEngine:

    def __init__(self):
        self.repo = CandleRepository()

    def evaluate(self, symbol: str, as_of) -> IntradayOverride:
        """
        Evaluate intraday volatility spike using 5-min candles
        """

        df = self.repo.get_5min_candles_upto(
            symbol,
            as_of,
            limit=ATR_LOOKBACK + ATR_PERIOD
        )

        if df.empty or len(df) < ATR_LOOKBACK:
            return IntradayOverride(
                active=False,
                reason="Insufficient intraday data",
                atr_value=0.0,
                percentile=0.0
            )

        # --- ATR calculation ---
        high = df["high"]
        low = df["low"]
        close = df["close"].shift(1)

        tr = pd.concat([
            high - low,
            (high - close).abs(),
            (low - close).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(ATR_PERIOD).mean().dropna()

        current_atr = float(atr.iloc[-1])
        percentile = float(
            np.percentile(atr,
                          (atr <= current_atr).mean() * 100)
        )

        # --- Decision ---
        if percentile >= ATR_SPIKE_PERCENTILE:
            return IntradayOverride(
                active=True,
                reason="5-min ATR volatility spike",
                atr_value=current_atr,
                percentile=round(percentile, 2)
            )

        return IntradayOverride(
            active=False,
            reason="Intraday volatility normal",
            atr_value=current_atr,
            percentile=round(percentile, 2)
        )

    def close(self):
        self.repo.close()
