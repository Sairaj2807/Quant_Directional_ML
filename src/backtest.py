def backtest_strategy(df, predictions):
    df = df.copy()
    df['signal'] = predictions.shift(1)
    df['strategy_ret'] = df['signal'] * df['ret_1']
    df['equity_curve'] = (1 + df['strategy_ret'].fillna(0)).cumprod()
    return df
