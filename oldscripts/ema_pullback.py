from strategies.base import BaseStrategy


class EMAPullbackStrategy(BaseStrategy):
    name = "EMA_PULLBACK"
    family = "TREND_FOLLOWING"

    def should_run(self, market_data) -> bool:
        return True  # cheap filter later

    def generate_signal(self, market_data):
        # placeholder logic
        if market_data["price_above_ema"]:
            return {"action": "BUY", "strategy": self.name}
        return None
