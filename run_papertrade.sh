#!/bin/bash

PROJECT_DIR="/home/ubuntu/sukesh/TradingSystemCopilot"

cd "$PROJECT_DIR" || exit 1

exec "$PROJECT_DIR/.venv/bin/python" nifty_short_straddle_papertrade.py >> cron.log 2>&1

