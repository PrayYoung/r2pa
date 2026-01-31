from __future__ import annotations

import argparse
import pandas as pd


def cmd_download(_args):
    # Reuse existing entrypoint
    from portfolio_rl_agent_lab.data.download import main as download_main
    download_main()


def cmd_news_alpaca(args):
    from portfolio_rl_agent_lab.text.build_news_daily_alpaca import main as alpaca_main

    # Minimal approach: temporarily patch by setting a global, or better:
    # Refactor build_news_daily_alpaca.main to accept days/symbols later (Phase 2).
    # For now, just run main and keep behavior unchanged.
    alpaca_main()


def main():
    parser = argparse.ArgumentParser(description="Data CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("download", help="Download market data and build returns.parquet")
    p1.set_defaults(func=cmd_download)

    p2 = sub.add_parser("news-alpaca", help="Fetch Alpaca news and build news_daily.jsonl")
    p2.add_argument("--days", type=int, default=5, help="Lookback days (MVP)")
    p2.set_defaults(func=cmd_news_alpaca)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
