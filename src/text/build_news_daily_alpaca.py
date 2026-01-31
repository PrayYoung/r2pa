from __future__ import annotations

import os
import pandas as pd

from src.text.news_alpaca import fetch_alpaca_news_daily, save_news_daily_jsonl


def main():
    os.makedirs("artifacts/data/processed", exist_ok=True)

    # Use your returns index as the canonical calendar
    rets = pd.read_parquet("artifacts/data/processed/returns.parquet")
    idx = pd.to_datetime(rets.index)

    # start_date = idx.min().strftime("%Y-%m-%d")
    end_date = idx.max().strftime("%Y-%m-%d")
    start_date = (idx.max() - pd.Timedelta(days=5)).strftime("%Y-%m-%d") # last 5 days for testing


    # Start with the same symbols you trade / have returns for.
    # If you want exact symbols, load them from returns columns instead.
    symbols = list(rets.columns)

    daily = fetch_alpaca_news_daily(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        include_content=False,  # start with headline+summary only
    )

    out_path = "artifacts/data/processed/news_daily.jsonl"
    save_news_daily_jsonl(daily, out_path=out_path, symbols=symbols)

    total_days = len(daily)
    total_bullets = sum(len(v) for v in daily.values())
    print(f"Saved: {out_path} | days={total_days} | bullets={total_bullets} | range={start_date}..{end_date}")


if __name__ == "__main__":
    main()
