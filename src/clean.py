import pandas as pd


def clean_data(df: pd.DataFrame):
    df = df.dropna()
    return df
