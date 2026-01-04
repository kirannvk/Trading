import csv
import datetime as dt
import json
import logging
import os
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from regime import classify_series, TREND, RANGE, EVENT

logger = logging.getLogger(__name__)


def apply_costs(pnl_per_unit: float, qty: int, brokerage_per_order: float, legs: int) -> float:
    # brokerage applied per leg per side (entry + exit). Total orders = legs * 2.
    total_orders = legs * 2
    return pnl_per_unit * qty - brokerage_per_order * total_orders


def vectorized_strangle(df_ce: pd.DataFrame, df_pe: pd.DataFrame, regimes: Dict[dt.date, str],
                        entry_time: dt.time, exit_time: dt.time, sl_pct: float,
                        brokerage_per_order: float, lot_size: int, allowed_regimes=None,
                        trade_log_path: Optional[str] = None, summary_path: Optional[str] = None,
                        equity_curve_path: Optional[str] = None) -> pd.DataFrame:
    if allowed_regimes is None:
        allowed_regimes = {RANGE}
    else:
        allowed_regimes = set(allowed_regimes)
    # Align by datetime
    df_ce = df_ce.copy()
    df_pe = df_pe.copy()
    df_ce['d'] = df_ce['date'].dt.date
    df_pe['d'] = df_pe['date'].dt.date

    records = []
    logger.info("Strangle backtest run: dates=%d regimes=%s entry=%s exit=%s sl_pct=%.2f allowed=%s",
                len(set(df_ce['d']).intersection(set(df_pe['d']))),
                list(set(regimes.values())), entry_time, exit_time, sl_pct, allowed_regimes)
    for d in sorted(set(df_ce['d']).intersection(set(df_pe['d']))):
        if regimes.get(d, RANGE) not in allowed_regimes:
            continue
        day_ce = df_ce[df_ce['d'] == d]
        day_pe = df_pe[df_pe['d'] == d]
        entry_row_ce = day_ce[day_ce['date'].dt.time >= entry_time].head(1)
        entry_row_pe = day_pe[day_pe['date'].dt.time >= entry_time].head(1)
        exit_row_ce = day_ce[day_ce['date'].dt.time >= exit_time].head(1)
        exit_row_pe = day_pe[day_pe['date'].dt.time >= exit_time].head(1)
        if entry_row_ce.empty or entry_row_pe.empty or exit_row_ce.empty or exit_row_pe.empty:
            logger.debug("Skip day %s due to missing entry/exit rows", d)
            continue
        ce_entry = entry_row_ce.iloc[0]
        pe_entry = entry_row_pe.iloc[0]
        ce_exit = exit_row_ce.iloc[0]
        pe_exit = exit_row_pe.iloc[0]

        ce_stop = ce_entry['close'] * (1 + sl_pct / 100)
        pe_stop = pe_entry['close'] * (1 + sl_pct / 100)

        ce_hit = (day_ce['high'] >= ce_stop).any()
        pe_hit = (day_pe['high'] >= pe_stop).any()

        ce_exit_price = ce_stop if ce_hit else ce_exit['close']
        pe_exit_price = pe_stop if pe_hit else pe_exit['close']

        pnl_per_unit = (ce_entry['close'] - ce_exit_price) + (pe_entry['close'] - pe_exit_price)
        pnl_after_cost = apply_costs(pnl_per_unit, lot_size, brokerage_per_order, legs=2)

        records.append({
            'date': d,
            'ce_entry': ce_entry['close'], 'pe_entry': pe_entry['close'],
            'ce_exit': ce_exit_price, 'pe_exit': pe_exit_price,
            'ce_hit_sl': bool(ce_hit), 'pe_hit_sl': bool(pe_hit),
            'pnl_per_unit': pnl_per_unit,
            'pnl_after_cost': pnl_after_cost
        })
        logger.debug("Day %s pnl_per_unit=%.2f pnl_after_cost=%.2f ce_hit=%s pe_hit=%s",
                     d, pnl_per_unit, pnl_after_cost, ce_hit, pe_hit)
    df_trades = pd.DataFrame(records)
    if trade_log_path:
        try:
            fields = ['date', 'ce_entry', 'pe_entry', 'ce_exit', 'pe_exit', 'ce_hit_sl', 'pe_hit_sl', 'pnl_per_unit', 'pnl_after_cost']
            write_header = not os.path.exists(trade_log_path)
            with open(trade_log_path, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields)
                if write_header:
                    w.writeheader()
                for r in records:
                    w.writerow(r)
        except Exception as e:
            logger.warning("Failed to write trade log: %s", e)
    if not df_trades.empty:
        eq = df_trades['pnl_after_cost'].cumsum()
        df_trades['equity_curve'] = eq
        if equity_curve_path:
            try:
                write_header = not os.path.exists(equity_curve_path)
                df_trades[['date', 'equity_curve']].to_csv(equity_curve_path, mode='a', header=write_header, index=False)
            except Exception as e:
                logger.warning("Failed to write equity curve: %s", e)
    if trade_log_path or summary_path:
        summary_data = summary(df_trades)
        target = summary_path or (os.path.splitext(trade_log_path)[0] + "_summary.json" if trade_log_path else None)
        if target:
            try:
                with open(target, 'w') as f:
                    json.dump(summary_data, f, default=str, indent=2)
            except Exception as e:
                logger.warning("Failed to write summary json: %s", e)
    logger.info("Strangle backtest produced %d trades", len(records))
    return df_trades


def summary(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        logger.warning("Summary requested on empty trades DataFrame")
        return {}
    total = df['pnl_after_cost'].sum()
    wins = (df['pnl_after_cost'] > 0).sum()
    losses = (df['pnl_after_cost'] < 0).sum()
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    equity = df['pnl_after_cost'].cumsum()
    max_eq = equity.cummax()
    drawdowns = max_eq - equity
    max_dd = drawdowns.max()
    avg_win = df.loc[df['pnl_after_cost'] > 0, 'pnl_after_cost'].mean() if wins else 0
    avg_loss = df.loc[df['pnl_after_cost'] < 0, 'pnl_after_cost'].mean() if losses else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss else float('inf')
    expectancy = (win_rate/100) * avg_win + ((100 - win_rate)/100) * avg_loss if (wins + losses) > 0 else 0
    logger.info("Summary total=%.2f win_rate=%.2f max_dd=%.2f trades=%d", total, win_rate, max_dd, len(df))
    return {
        'total_pnl': total,
        'win_rate_pct': win_rate,
        'max_drawdown': max_dd,
        'trades': len(df),
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
    }
