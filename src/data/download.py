import os
import yfinance as yf
import pandas as pd
from src.config import CFG

def download_prices(tickers, start, end) -> pd.DataFrame:
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    # auto_adjust=True 会把分红拆股调整进价格
    prices = df["Close"].copy()
    prices = prices.dropna(how="all")
    prices = prices.ffill().dropna()
    return prices

def main():
    os.makedirs("data/raw", exist_ok=True)
    prices = download_prices(CFG.tickers, CFG.start, CFG.end)
    path = "data/raw/prices.parquet"
    prices.to_parquet(path)
    print(f"Saved: {path} | shape={prices.shape} | from {prices.index.min()} to {prices.index.max()}")

if __name__ == "__main__":
    main()
