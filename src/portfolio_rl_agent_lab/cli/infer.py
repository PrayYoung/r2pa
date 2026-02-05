from __future__ import annotations

import argparse

import json

from portfolio_rl_agent_lab.infer.run_policy import run_policy, _parse_weights, _parse_tickers


def cmd_run(args):
    cw = _parse_weights(args.current_weights) if args.current_weights else None
    tickers = _parse_tickers(args.tickers) if args.tickers else None
    result = run_policy(
        model_path=args.model,
        algo=args.algo,
        returns_path=args.returns,
        asof=args.asof,
        current_weights=cw,
        use_regime=not args.no_regime,
        regime_source=args.regime_source,
        out_path=args.out,
        live_yahoo=args.live_yahoo,
        tickers=tickers,
        lookback_days=args.lookback_days,
        live_news=args.live_news,
        news_lookback_days=args.news_lookback_days,
        news_include_content=args.news_include_content,
    )
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Inference CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="Run single-date inference")
    p.add_argument("--model", default="artifacts/models/ppo_portfolio", help="Path to trained RL model")
    p.add_argument("--algo", choices=["ppo", "a2c", "sac", "td3"], default="ppo", help="RL algorithm type")
    p.add_argument("--returns", default="artifacts/data/processed/returns.parquet", help="Returns parquet")
    p.add_argument("--asof", default=None, help="YYYY-MM-DD (default: last date in returns)")
    p.add_argument("--current-weights", default=None, help="Comma list of current weights")
    p.add_argument("--no-regime", action="store_true", help="Disable regime features")
    p.add_argument("--regime-source", default="heuristic", help="Regime source: heuristic|local|student")
    p.add_argument("--out", default=None, help="Write output JSON to path")
    p.add_argument("--live-yahoo", action="store_true", help="Fetch latest prices from Yahoo for inference")
    p.add_argument("--tickers", default=None, help="Comma list of tickers (defaults to CFG.tickers)")
    p.add_argument("--lookback-days", type=int, default=180, help="Yahoo lookback days (must cover window)")
    p.add_argument("--live-news", action="store_true", help="Fetch live news for local LLM regime")
    p.add_argument("--news-lookback-days", type=int, default=5, help="News lookback days")
    p.add_argument("--news-include-content", action="store_true", help="Include full news content when available")
    p.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
