import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_data(ticker, start_date="2015-01-01"):
    file_path = DATA_DIR / f"{ticker}.csv"

    # Load cached data
    if file_path.exists():
        df = pd.read_csv(file_path, parse_dates=["Date"])
        return df

    # Download
    df = yf.download(ticker, start=start_date, progress=False)
    df.reset_index(inplace=True)

    # Safety for yfinance multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Date','Open','High','Low','Close','Volume']]
    df = df.sort_values("Date").dropna().reset_index(drop=True)

    df.to_csv(file_path, index=False)
    return df


def add_returns(df):
    df = df.copy()
    df['ret_1'] = df['Close'].pct_change()
    df['log_ret_1'] = np.log(df['Close'] / df['Close'].shift(1))
    return df
