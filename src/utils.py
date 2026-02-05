import numpy as np
import pandas as pd

def set_seed(seed=42):
    np.random.seed(seed)


def check_nan_inf(df):
    """Quick sanity check before modeling"""
    nan_count = df.isna().sum().sum()
    inf_count = np.isinf(df.select_dtypes(include=[float, int])).sum().sum()
    return {"nan": nan_count, "inf": inf_count}


def print_df_info(df, name="DataFrame"):
    print(f"\n{name} shape: {df.shape}")
    print(df.head())
