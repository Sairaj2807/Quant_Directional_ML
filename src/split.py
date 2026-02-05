def chronological_split(df, split_ratio=0.8):
    split = int(len(df) * split_ratio)
    train = df.iloc[:split]
    test = df.iloc[split:]
    return train, test
