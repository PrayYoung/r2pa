from __future__ import annotations

import argparse


def cmd_train(_args):
    from portfolio_rl_agent_lab.train import main as m
    m()


def cmd_backtest(_args):
    from portfolio_rl_agent_lab.eval.backtest import main as m
    m()


def cmd_benchmarks(_args):
    from portfolio_rl_agent_lab.eval.benchmarks import main as m
    m()


def cmd_diagnostics(_args):
    from portfolio_rl_agent_lab.eval.diagnostics import main as m
    m()


def main():
    parser = argparse.ArgumentParser(description="RL CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("train").set_defaults(func=cmd_train)
    sub.add_parser("backtest").set_defaults(func=cmd_backtest)
    sub.add_parser("benchmarks").set_defaults(func=cmd_benchmarks)
    sub.add_parser("diagnostics").set_defaults(func=cmd_diagnostics)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
