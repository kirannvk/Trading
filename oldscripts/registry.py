from strategies.trend_following.ema_pullback import EMAPullbackStrategy
from strategies.breakout.opening_range_breakout import OpeningRangeBreakoutStrategy
from strategies.mean_reversion.vwap_reversion import VWAPReversionStrategy


class StrategyRegistry:
    """
    Central registry for all strategies
    """

    def __init__(self):
        self._strategies = [
            EMAPullbackStrategy(),
            OpeningRangeBreakoutStrategy(),
            VWAPReversionStrategy(),
        ]

    def get_strategies_by_family(self, family: str):
        return [
            s for s in self._strategies
            if s.family == family
        ]

    def get_all(self):
        return self._strategies
