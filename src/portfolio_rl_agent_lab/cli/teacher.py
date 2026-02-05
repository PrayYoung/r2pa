from __future__ import annotations

import argparse


def cmd_build_local(args):
    from portfolio_rl_agent_lab.llm.build_regime_features_local import build_regime_features_local
    build_regime_features_local(max_steps=args.max_steps)


def main():
    parser = argparse.ArgumentParser(description="Teacher (oracle) CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-local", help="Build LLM teacher regime features (local LLM)")
    p.add_argument("--max-steps", type=int, default=5, help="Limit steps for quick runs (default: 5)")
    p.set_defaults(func=cmd_build_local)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
