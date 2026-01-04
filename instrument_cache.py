import pandas as pd
import datetime as dt
from typing import Optional, Dict, Any

from kite_trade import KiteApp


INDEX_TOKENS = {
    "NIFTY": "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK",
}

STRIKE_STEP = {
    "NIFTY": 50,
    "BANKNIFTY": 100,
}


class InstrumentCache:
    def __init__(self, nfo: pd.DataFrame, nse: pd.DataFrame):
        self.nfo = nfo
        self.nse = nse

    @classmethod
    def from_files_or_api(cls, kite: KiteApp, nfo_path: str = "nso_instruments.csv", nse_path: str = "nse_instruments.csv"):
        """Always fetch instruments from Kite API; raise a clear error if unavailable."""
        try:
            nfo_df = pd.DataFrame(kite.instruments("NFO"))
            nse_df = pd.DataFrame(kite.instruments("NSE"))
        except Exception as e:
            raise RuntimeError(f"Failed to fetch instruments from Kite: {e}") from e
        return cls(nfo_df, nse_df)

    def get_index_token(self, underlying: str) -> int:
        row = self.nse[(self.nse['segment'] == 'INDICES') & (self.nse['tradingsymbol'] == INDEX_TOKENS[underlying])]
        if row.empty:
            raise RuntimeError(f"Index token not found for {underlying}")
        return int(row.iloc[0]['instrument_token'])

    def get_nearest_expiry(self, underlying: str, weekly_first: bool = True) -> dt.date:
        df = self.nfo[(self.nfo['instrument_type'].isin(['CE', 'PE'])) & (self.nfo['tradingsymbol'].str.startswith(underlying))]
        if df.empty:
            raise RuntimeError(f"No options found for {underlying}")
        expiries = sorted(df['expiry'].dropna().unique())
        if weekly_first:
            return pd.to_datetime(expiries[0]).date()
        return pd.to_datetime(expiries[0]).date()

    def round_to_strike(self, price: float, underlying: str) -> float:
        step = STRIKE_STEP.get(underlying, 100)
        return round(price / step) * step

    def find_option(self, underlying: str, expiry: dt.date, strike: float, opt_type: str) -> Optional[Dict[str, Any]]:
        df = self.nfo[
            (self.nfo['instrument_type'] == opt_type) &
            (self.nfo['tradingsymbol'].str.startswith(underlying)) &
            (pd.to_datetime(self.nfo['expiry']).dt.date == expiry) &
            (self.nfo['strike'] == strike)
        ]
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def get_spot_from_ws(self, kite: KiteApp, underlying: str, api_key: str, enctoken: str, user_id: str) -> float:
        token = self.get_index_token(underlying)
        # TODO: update ltp_ws.stream_ltps to return the last tick; for now, attempt REST LTP fallback via token
        try:
            resp = kite.session.get(f"{kite.root_url}/quote/ltp", params={"i": f"NSE:{token}"}, headers=kite.headers).json()
            ltp = resp.get("data", {}).get(f"NSE:{token}", {}).get("last_price", 0)
            return ltp or 0.0
        except Exception:
            return 0.0

