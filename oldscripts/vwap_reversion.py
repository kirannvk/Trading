from strategies.base import BaseStrategy


class VWAPReversionStrategy(BaseStrategy):
    name = "VWAP_REVERSION"
    family = "MEAN_REVERSION"

    def should_run(self, market_data) -> bool:
        return True

    def generate_signal(self, market_data):
        if market_data["price_far_from_vwap"]:
            return {"action": "SELL", "strategy": self.name}
        return None
