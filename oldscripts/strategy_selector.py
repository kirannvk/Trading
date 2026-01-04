from dataclasses import dataclass
from typing import List

from market_intel.regime_engine import MarketRegime


@dataclass
class StrategyDecision:
    allowed_strategies: List[str]
    risk_mode: str               # AGGRESSIVE / NORMAL / CONSERVATIVE / OFF
    reason: str


class StrategySelector:

    def select(self, regime: MarketRegime) -> StrategyDecision:
        """
        Decide which strategy families are allowed based on MarketRegime
        """

        # Hard veto
        if not regime.tradable:
            return StrategyDecision(
                allowed_strategies=["NO_TRADE"],
                risk_mode="OFF",
                reason="Market not tradable (time/volatility veto)"
            )

        # Volatility veto
        if regime.volatility == "HIGH":
            return StrategyDecision(
                allowed_strategies=["NO_TRADE"],
                risk_mode="OFF",
                reason="High volatility regime"
            )

        # Sideways market
        if regime.trend == "SIDEWAYS":
            return StrategyDecision(
                allowed_strategies=[
                    "MEAN_REVERSION",
                    "OPTION_SELLING"
                ],
                risk_mode="CONSERVATIVE",
                reason="Sideways market with controlled volatility"
            )

        # Trending markets
        if regime.trend in ("UP", "DOWN"):
            if regime.volatility == "LOW":
                return StrategyDecision(
                    allowed_strategies=[
                        "TREND_FOLLOWING",
                        "BREAKOUT"
                    ],
                    risk_mode="AGGRESSIVE",
                    reason="Strong trend with low volatility"
                )

            return StrategyDecision(
                allowed_strategies=[
                    "TREND_FOLLOWING"
                ],
                risk_mode="NORMAL",
                reason="Trend present with normal volatility"
            )

        # Fallback (should never happen)
        return StrategyDecision(
            allowed_strategies=["NO_TRADE"],
            risk_mode="OFF",
            reason="Undefined regime state"
        )
