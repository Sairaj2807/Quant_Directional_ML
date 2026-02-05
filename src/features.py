from pathlib import Path
import pandas as pd
import numpy as np
import pandas_ta as ta

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def save_features(df, ticker):
    df.to_csv(PROCESSED_DIR / f"{ticker}_features.csv", index=False)


def add_lagged_returns(df):
    df = df.copy()
    for lag in [1,5,10,20]:
        df[f'ret_{lag}'] = df['Close'].pct_change(lag)
        
    return df   

def add_moving_averages(df):
    df = df.copy()

    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]  # force Series

    for window in [5,10,20,50,100,200]:
        sma = close.rolling(window=window).mean()
        df[f'sma_{window}'] = sma
        df[f'price_sma_{window}'] = (close / sma) - 1

    return df


def add_volatility(df):
    df = df.copy()
    for window in [10, 20, 60]:
        df[f'vol_{window}'] = df['ret_1'].rolling(window=window).std()
        
    return df

def add_momentum(df):
    df = df.copy()
    df['roc_10'] = df['Close'].pct_change(10)
    df['roc_20'] = df['Close'].pct_change(20)
    
    return df

def add_technical_indicators(df):
    df = df.copy()
    df['rsi_14'] = ta.rsi(df['Close'], length=14)
    
    macd = ta.macd(df['Close'])
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    
    bb = ta.bbands(df['Close'], length=20)
    df['bb_width'] = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / df['Close']
    
    return df

def add_calendar_features(df):
    df = df.copy()

    if 'Date' in df.columns:
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
    else:
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month

    return df
