import pandas as pd
from auth_utils import load_config, generate_enctoken
from kite_trade import KiteApp

# Load credentials from config.yaml (auth.user_id/password/totp_secret)
cfg = load_config()
enctoken = generate_enctoken(cfg)
kite = KiteApp(enctoken)

# Fetch instruments
nfo_df = pd.DataFrame(kite.instruments("NFO"))
nse_df = pd.DataFrame(kite.instruments("NSE"))
all_df = pd.DataFrame(kite.instruments())

# Persist to CSV
nfo_df.to_csv("nso_instruments.csv", index=False)
nse_df.to_csv("nse_instruments.csv", index=False)
all_df.to_csv("all_instruments.csv", index=False)

print("Saved instruments: nso_instruments.csv, nse_instruments.csv, all_instruments.csv")
