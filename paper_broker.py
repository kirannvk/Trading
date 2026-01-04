import pandas as pd
from typing import Dict, Any, List
import logging


logger = logging.getLogger(__name__)


class PaperBroker:
    def __init__(self, slippage_ticks: float = 1.0, tick_size: float = 0.05, brokerage_per_order: float = 40):
        self.slippage_ticks = slippage_ticks
        self.tick_size = tick_size
        self.brokerage_per_order = brokerage_per_order
        self.trades: List[Dict[str, Any]] = []

    def _fill(self, ltp: float, side: str) -> float:
        price = ltp + self.slippage_ticks * self.tick_size * 0.5 if side == "BUY" else ltp - self.slippage_ticks * self.tick_size * 0.5
        logger.debug("Fill side=%s ltp=%.2f price=%.2f", side, ltp, price)
        return price

    def enter(self, symbol: str, side: str, ltp: float, qty: int, timestamp) -> Dict[str, Any]:
        price = self._fill(ltp, side)
        trade = {
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'price': price,
            'timestamp': timestamp,
            'type': 'ENTRY'
        }
        self.trades.append(trade)
        logger.info("ENTRY %s side=%s qty=%d price=%.2f ts=%s", symbol, side, qty, price, timestamp)
        return trade

    def exit(self, symbol: str, side: str, ltp: float, qty: int, timestamp) -> Dict[str, Any]:
        price = self._fill(ltp, side)
        trade = {
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'price': price,
            'timestamp': timestamp,
            'type': 'EXIT'
        }
        self.trades.append(trade)
        logger.info("EXIT %s side=%s qty=%d price=%.2f ts=%s", symbol, side, qty, price, timestamp)
        return trade

    def costs(self, legs: int) -> float:
        c = self.brokerage_per_order * legs
        logger.debug("Costs legs=%d cost=%.2f", legs, c)
        return c

    def pnl(self) -> float:
        # simplistic: assumes paired entry/exit per leg
        pnl = 0.0
        symbol_groups = {}
        for t in self.trades:
            symbol_groups.setdefault(t['symbol'], []).append(t)
        for sym, trades in symbol_groups.items():
            entries = [t for t in trades if t['type'] == 'ENTRY']
            exits = [t for t in trades if t['type'] == 'EXIT']
            if not entries or not exits:
                continue
            entry_avg = sum(t['price'] for t in entries) / len(entries)
            exit_avg = sum(t['price'] for t in exits) / len(exits)
            qty = sum(t['qty'] for t in entries)
            if entries[0]['side'] == 'BUY':
                pnl += (exit_avg - entry_avg) * qty
            else:
                pnl += (entry_avg - exit_avg) * qty
        pnl -= self.costs(len(self.trades))
        logger.info("PaperBroker PnL=%.2f after costs", pnl)
        return pnl
