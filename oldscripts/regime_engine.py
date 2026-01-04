from dataclasses import dataclass
from datetime import datetime, time
import pytz

from db.candle_repository import CandleRepository
from db.market_regime_repository import MarketRegimeRepository

from market_intel.regime_logic import (
    classify_trend,
    classify_volatility
)

from config.settings import (
    MARKET_START_TRADE_TIME,
    MARKET_STOP_TRADE_TIME
)

IST = pytz.timezone("Asia/Kolkata")


@dataclass
class MarketRegime:
    symbol: str
    as_of: datetime

    trend: str
    trend_confidence: float

    volatility: str
    volatility_confidence: float

    tradable: bool
    overall_confidence: float


class MarketRegimeEngine:
    def __init__(self):
        self.candle_repo = CandleRepository()
        self.regime_repo = MarketRegimeRepository()

    def _is_market_time(self, ts: datetime) -> bool:
        ts_ist = ts.astimezone(IST)
        t = ts_ist.time()

        start = time(*MARKET_START_TRADE_TIME)
        stop = time(*MARKET_STOP_TRADE_TIME)

        return start <= t <= stop

    def evaluate(self, symbol: str, as_of: datetime | None = None, persist: bool = True) -> MarketRegime:
        if as_of is None:
            as_of = datetime.now(tz=IST)

        # --- Fetch daily data upto as_of ---
        daily_df = self.candle_repo.get_daily_candles_upto(
            symbol, as_of, limit=300
        )

        if daily_df.empty:
            raise ValueError("No daily data available")

        # --- Regimes ---
        trend_regime = classify_trend(daily_df)
        vol_regime = classify_volatility(daily_df)

        # --- Tradability gate ---
        tradable = True

        if not self._is_market_time(as_of):
            tradable = False

        if vol_regime.volatility == "HIGH":
            tradable = False

        overall_confidence = round(
            (trend_regime.confidence + vol_regime.confidence) / 2, 2
        )

        regime_snapshot = MarketRegime(
            symbol=symbol,
            as_of=as_of,
            trend=trend_regime.trend,
            trend_confidence=float(trend_regime.confidence),
            volatility=vol_regime.volatility,
            volatility_confidence=float(vol_regime.confidence),
            tradable=tradable,
            overall_confidence=float(overall_confidence)
        )

        if persist:
            print("Saving regime snapshot:", regime_snapshot)
            self.regime_repo.save(regime_snapshot)

        # --- Persist snapshot ---

        return regime_snapshot

    def close(self):
        self.candle_repo.close()
        self.regime_repo.close()
