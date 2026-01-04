import time
import logging
from typing import List

from kiteconnect import KiteTicker
from auth_utils import load_config, get_kite_with_refresh
from main import get_index_tokens

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def stream_ltps(api_key: str, enctoken: str, user_id: str, instrument_tokens: List[int], duration_sec: int = 15):
    kws = KiteTicker(api_key=api_key, access_token=f"{enctoken}&user_id={user_id}")

    def on_ticks(ws, ticks):
        for t in ticks:
            logger.info("Tick: token=%s ltp=%s", t.get("instrument_token"), t.get("last_price"))

    def on_connect(ws, response):
        logger.info("WebSocket connected, subscribing to %s", instrument_tokens)
        ws.subscribe(instrument_tokens)
        ws.set_mode(ws.MODE_QUOTE, instrument_tokens)

    kws.on_ticks = on_ticks
    kws.on_connect = on_connect

    logger.info("Connecting WebSocket for LTP stream...")
    kws.connect(threaded=True)
    timeout = time.time() + duration_sec
    while not kws.is_connected() and time.time() < timeout:
        time.sleep(0.5)
    if not kws.is_connected():
        logger.error("WebSocket failed to connect within %s seconds", duration_sec)
        return
    logger.info("WebSocket: Connected")
    time.sleep(duration_sec)
    try:
        kws.unsubscribe(instrument_tokens)
        logger.info("Unsubscribed and closing")
    except Exception:
        pass
    kws.close()


def main():
    cfg = load_config()
    kite, enctoken = get_kite_with_refresh(cfg)
    profile = kite.profile()
    user_id = profile.get("user_id") if profile else None
    if not user_id:
        raise RuntimeError("Could not fetch user_id from profile; ensure credentials are correct")
    api_key = cfg.get("auth", {}).get("api_key", "AlgoTrader")

    # Auto-discover index tokens from NSE instruments if not provided
    tokens_cfg = cfg.get("ltp_tokens")
    if tokens_cfg:
        tokens = tokens_cfg
    else:
        nse = kite.instruments("NSE")
        underlying = cfg.get("strategy", {}).get("underlying", "BANKNIFTY")
        tokens_map = get_index_tokens(nse)
        tok = tokens_map.get(underlying.upper())
        if not tok:
            raise RuntimeError(f"No index token found for {underlying}")
        tokens = [tok]
    stream_ltps(api_key, enctoken, user_id, tokens, duration_sec=30)


if __name__ == "__main__":
    main()
