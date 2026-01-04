import os
import yaml
import pyotp
import logging
from typing import Any, Dict, Tuple
from kite_trade import get_enctoken, KiteApp

CONFIG_PATH = os.getenv("TRADER_CONFIG", "config.yaml")
ENCTOKEN_CACHE = ".enctoken"
logger = logging.getLogger(__name__)


def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def generate_enctoken(cfg: Dict[str, Any]) -> str:
    auth = cfg.get("auth", {})
    user_id = auth.get("user_id")
    password = auth.get("password")
    totp_secret = auth.get("totp_secret")
    if not (user_id and password and totp_secret):
        raise RuntimeError("auth.user_id/password/totp_secret must be set in config.yaml")
    totp = pyotp.TOTP(totp_secret)
    twofa = totp.now()
    return get_enctoken(user_id, password, twofa)


def read_cached_enctoken() -> str:
    if os.path.exists(ENCTOKEN_CACHE):
        try:
            with open(ENCTOKEN_CACHE, "r") as f:
                tok = f.read().strip()
                if tok:
                    return tok
        except Exception:
            pass
    return ""


def write_cached_enctoken(token: str):
    try:
        with open(ENCTOKEN_CACHE, "w") as f:
            f.write(token)
    except Exception as e:
        logger.warning("Failed to write enctoken cache: %s", e)


def get_kite_with_refresh(cfg: Dict[str, Any]) -> Tuple[KiteApp, str]:
    # Always regenerate a fresh token per run; keep one retry if validation fails
    fresh = generate_enctoken(cfg)
    write_cached_enctoken(fresh)
    kite = KiteApp(fresh)
    try:
        prof = kite.profile()
        if prof:
            return kite, fresh
        logger.warning("Fresh enctoken returned None profile; regenerating once")
    except Exception as e:
        logger.warning("Fresh enctoken validation failed: %s; regenerating once", e)

    # One more retry
    fresh = generate_enctoken(cfg)
    write_cached_enctoken(fresh)
    kite = KiteApp(fresh)
    prof = kite.profile()
    if not prof:
        raise RuntimeError("Profile returned None after second refresh; credentials may be invalid")
    return kite, fresh

