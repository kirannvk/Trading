from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """
    Base class for all strategies
    """

    name: str
    family: str

    @abstractmethod
    def should_run(self, market_data) -> bool:
        """
        Quick check (cheap) before full evaluation
        """
        pass

    @abstractmethod
    def generate_signal(self, market_data):
        """
        Returns a signal or None
        """
        pass
