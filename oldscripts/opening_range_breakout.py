from strategies.base import BaseStrategy


class OpeningRangeBreakoutStrategy(BaseStrategy):
    name = "OPENING_RANGE_BREAKOUT"
    family = "BREAKOUT"

    def should_run(self, market_data) -> bool:
        return market_data["is_opening_range_done"]

    def generate_signal(self, market_data):
        if market_data["breakout"]:
            return {"action": "BUY", "strategy": self.name}
        return None
