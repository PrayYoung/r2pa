from __future__ import annotations

import pandas as pd


def load_returns(path: str = "artifacts/data/processed/returns.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


def load_news_features(path: str = "artifacts/data/processed/news_features.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()
