import streamlit as st
import pandas as pd
import json
import os

TRADES_PATH = os.getenv("BACKTEST_TRADES", "backtest_trades.csv")
SUMMARY_PATH = os.getenv("BACKTEST_SUMMARY", "backtest_summary.json")
PAPER_TRADES_PATH = os.getenv("PAPER_TRADES", "paper_trades.csv")
BACKTEST_EQUITY_PATH = os.getenv("BACKTEST_EQUITY", "backtest_equity_curve.csv")

st.set_page_config(page_title="Trading Dashboard", layout="wide")
st.title("Trading Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.header("Backtest Trades")
    if os.path.exists(TRADES_PATH):
        df = pd.read_csv(TRADES_PATH)
        st.dataframe(df.tail(100))
        st.metric("Trades", len(df))
        st.metric("Total PnL", df['pnl_after_cost'].sum() if 'pnl_after_cost' in df else 0)
    else:
        st.info(f"No backtest trades file at {TRADES_PATH}")

with col2:
    st.header("Backtest Summary")
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, 'r') as f:
            summ = json.load(f)
        st.json(summ)
        if os.path.exists(BACKTEST_EQUITY_PATH):
            eq_df = pd.read_csv(BACKTEST_EQUITY_PATH)
            if not eq_df.empty:
                st.line_chart(eq_df.set_index('date')['equity_curve'])
    else:
        st.info(f"No summary file at {SUMMARY_PATH}")

st.header("Paper Trades")
if os.path.exists(PAPER_TRADES_PATH):
    dfp = pd.read_csv(PAPER_TRADES_PATH)
    st.dataframe(dfp.tail(100))
    if 'price' in dfp and 'side' in dfp and 'symbol' in dfp:
        # Simple PnL proxy for display only
        st.metric("Paper trades", len(dfp))
else:
    st.info(f"No paper trades file at {PAPER_TRADES_PATH}")

st.caption("Set BACKTEST_TRADES, BACKTEST_SUMMARY, BACKTEST_EQUITY, PAPER_TRADES env vars to point to custom files.")
