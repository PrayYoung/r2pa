from __future__ import annotations

import os
import pandas as pd

from portfolio_rl_agent_lab.text.news_alpaca import fetch_alpaca_news_daily, save_news_daily_jsonl


def build_news_daily_alpaca(
    returns_path: str = "artifacts/data/processed/returns.parquet",
    out_path: str = "artifacts/data/processed/news_daily.jsonl",
    lookback_days: int = 5,
    include_content: bool = False,
):
    os.makedirs("artifacts/data/processed", exist_ok=True)

    # Use your returns index as the canonical calendar
    rets = pd.read_parquet(returns_path)
    idx = pd.to_datetime(rets.index)

    # start_date = idx.min().strftime("%Y-%m-%d")
    end_date = idx.max().strftime("%Y-%m-%d")
    start_date = (idx.max() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d") # last N days for testing

    # Start with the same symbols you trade / have returns for.
    # If you want exact symbols, load them from returns columns instead.
    symbols = list(rets.columns)

    daily = fetch_alpaca_news_daily(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        include_content=include_content,  # start with headline+summary only
    )

    save_news_daily_jsonl(daily, out_path=out_path, symbols=symbols)

    total_days = len(daily)
    total_bullets = sum(len(v) for v in daily.values())
    print(f"Saved: {out_path} | days={total_days} | bullets={total_bullets} | range={start_date}..{end_date}")
    return daily

def main():
    build_news_daily_alpaca()

if __name__ == "__main__":
    main()
