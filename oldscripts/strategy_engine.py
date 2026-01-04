from market_intel.strategy_selector import StrategySelector
from db.strategy_decision_repository import StrategyDecisionRepository


class StrategyEngine:
    def __init__(self):
        self.selector = StrategySelector()
        self.repo = StrategyDecisionRepository()

    def evaluate(self, regime, persist=True):
        decision = self.selector.select(regime)

        if persist:
            self.repo.save(
                symbol=regime.symbol,
                as_of=regime.as_of,
                decision=decision
            )

        return decision

    def close(self):
        self.repo.close()
