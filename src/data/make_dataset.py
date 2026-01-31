import os
import numpy as np
import pandas as pd
from src.config import CFG

def load_prices(path="artifacts/data/raw/prices.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)

def make_returns(prices: pd.DataFrame) -> pd.DataFrame:
    rets = prices.pct_change().dropna()
    return rets

def main():
    os.makedirs("artifacts/data/processed", exist_ok=True)
    prices = load_prices()
    rets = make_returns(prices)

    rets.to_parquet("artifacts/data/processed/returns.parquet")
    print(f"Saved returns: artifacts/data/processed/returns.parquet | shape={rets.shape}")

if __name__ == "__main__":
    main()
