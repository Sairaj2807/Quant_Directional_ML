def create_target(df, horizon=15):
    df = df.copy()
    df['target'] = (df['Close'].shift(-horizon) > df['Close']).astype(int)
    return df.iloc[:-horizon]
