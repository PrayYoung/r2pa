import pandas as pd
from src.config import CFG

def load_regime_store(path: str = None) -> pd.DataFrame:
    path = path or CFG.regime_store_path
    df = pd.read_parquet(path)
    # index should be datetime-like
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df

def save_regime_store(df: pd.DataFrame, path: str = None):
    path = path or CFG.regime_store_path
    df.sort_index().to_parquet(path)
