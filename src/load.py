import pandas as pd


# load the data from the data directory
def load_data(data_dir: str) -> pd.DataFrame:
    return pd.read_csv(data_dir)
