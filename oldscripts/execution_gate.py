from dataclasses import dataclass
from typing import List

from market_intel.regime_engine import MarketRegimeEngine
from market_intel.strategy_engine import StrategyEngine
from market_intel.intraday_override import IntradayVolatilityOverrideEngine


@dataclass
class ExecutionDecision:
    can_trade: bool
    allowed_strategies: List[str]
    risk_mode: str
    reason: str


class ExecutionGate:
    """
    Final authority that decides whether trading is allowed
    and which strategy families may execute.
    """

    def __init__(self):
        self.regime_engine = MarketRegimeEngine()
        self.strategy_engine = StrategyEngine()
        self.override_engine = IntradayVolatilityOverrideEngine()

    def evaluate(self, symbol: str, as_of, persist=True) -> ExecutionDecision:
        # --- 1. Market regime (daily) ---
        regime = self.regime_engine.evaluate(
            symbol,
            as_of=as_of,
            persist=persist
        )

        if not regime.tradable:
            return ExecutionDecision(
                can_trade=False,
                allowed_strategies=[],
                risk_mode="OFF",
                reason="Market regime marked as non-tradable"
            )

        # --- 2. Strategy selector ---
        strategy_decision = self.strategy_engine.evaluate(
            regime,
            persist=persist
        )

        if strategy_decision.risk_mode == "OFF":
            return ExecutionDecision(
                can_trade=False,
                allowed_strategies=[],
                risk_mode="OFF",
                reason=strategy_decision.reason
            )

        # --- 3. Intraday volatility override ---
        override = self.override_engine.evaluate(symbol, as_of)

        if override.active:
            return ExecutionDecision(
                can_trade=False,
                allowed_strategies=[],
                risk_mode="OFF",
                reason=f"Intraday override active: {override.reason}"
            )

        # --- Final approval ---
        return ExecutionDecision(
            can_trade=True,
            allowed_strategies=strategy_decision.allowed_strategies,
            risk_mode=strategy_decision.risk_mode,
            reason="All gates passed"
        )

    def close(self):
        self.regime_engine.close()
        self.strategy_engine.close()
        self.override_engine.close()
