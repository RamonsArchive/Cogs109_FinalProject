import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame):
    df = df.dropna()
    return df


def log_transform_data(df: pd.DataFrame, target_column: str):
    """Log transform the TARGET for linear regression"""
    df = df.copy()  # Don't modify original

    # Use log1p: log(x + 1) to handle zeros naturally
    df[target_column] = np.log1p(df[target_column])

    print(f"✅ Log-transformed '{target_column}' using log1p(x)")
    print(f"   Original range: [0, {int(np.exp(df[target_column].max()) - 1)}]")
    print(
        f"   Log range: [{df[target_column].min():.3f}, {df[target_column].max():.3f}]"
    )

    return df
