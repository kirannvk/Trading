import datetime as dt
import pandas as pd
from regime import classify_day, classify_series, TREND, RANGE, EVENT


def _make_day(ohlc):
    rows = []
    base_date = dt.datetime(2024, 1, 1, 9, 15)
    for i, (o, h, l, c) in enumerate(ohlc):
        rows.append({
            'date': base_date + dt.timedelta(minutes=5*i),
            'open': o, 'high': h, 'low': l, 'close': c, 'volume': 0
        })
    return pd.DataFrame(rows)


def test_classify_day_event_gap():
    day = _make_day([(100, 101, 99, 100)])
    regime = classify_day(day, prev_close=90, vix_change=0, gap_threshold=1)
    assert regime == EVENT


def test_classify_day_trend_breakout():
    day = _make_day([
        (100, 101, 99, 100),
        (100, 103, 99, 103),  # breakout above ORH
    ])
    regime = classify_day(day, prev_close=100, vix_change=0, opening_range_minutes=5)
    assert regime == TREND


def test_classify_day_range_default():
    day = _make_day([
        (100, 101, 99, 100),
        (100, 100.5, 99.5, 100.2),
    ])
    regime = classify_day(day, prev_close=100, vix_change=0)
    assert regime == RANGE


def test_classify_series_counts():
    day1 = _make_day([(100, 101, 99, 100)])
    day1['date'] = day1['date']
    day2 = _make_day([(105, 106, 104, 106)])
    day2['date'] = day2['date'] + dt.timedelta(days=1)
    df = pd.concat([day1, day2], ignore_index=True)
    regimes = classify_series(df, opening_range_minutes=30, gap_threshold=0.5)
    assert len(regimes) == 2
    assert set(regimes.values()).issubset({TREND, RANGE, EVENT})
