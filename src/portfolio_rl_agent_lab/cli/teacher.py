from __future__ import annotations

import argparse


def cmd_build_local(_args):
    from portfolio_rl_agent_lab.llm.build_regime_features_local import main as build_main
    build_main()


def main():
    parser = argparse.ArgumentParser(description="Teacher (oracle) CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-local", help="Build LLM teacher regime features (local LLM)")
    p.add_argument("--days", type=int, default=5, help="Lookback days (MVP)")
    p.set_defaults(func=cmd_build_local)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
