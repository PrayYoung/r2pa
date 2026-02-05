from __future__ import annotations

import argparse


def cmd_train(args):
    from portfolio_rl_agent_lab.train.train_rl import train_rl
    train_rl(algo=args.algo, total_timesteps=args.timesteps)


def cmd_backtest(args):
    from portfolio_rl_agent_lab.eval.backtest import run_backtest
    model_path = args.model or f"artifacts/models/{args.algo}_portfolio"
    run_backtest(algo=args.algo, model_path=model_path)


def cmd_benchmarks(args):
    from portfolio_rl_agent_lab.eval.benchmarks import run_benchmarks
    model_path = args.model or f"artifacts/models/{args.algo}_portfolio"
    run_benchmarks(algo=args.algo, model_path=model_path)


def cmd_diagnostics(args):
    from portfolio_rl_agent_lab.eval.diagnostics import run_diagnostics
    model_path = args.model or f"artifacts/models/{args.algo}_portfolio"
    run_diagnostics(algo=args.algo, model_path=model_path)


def main():
    parser = argparse.ArgumentParser(description="RL CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--algo", choices=["ppo", "a2c", "sac", "td3"], default="ppo")
    p_train.add_argument("--timesteps", type=int, default=200_000)
    p_train.set_defaults(func=cmd_train)

    p_backtest = sub.add_parser("backtest")
    p_backtest.add_argument("--algo", choices=["ppo", "a2c", "sac", "td3"], default="ppo")
    p_backtest.add_argument("--model", default=None)
    p_backtest.set_defaults(func=cmd_backtest)

    p_bench = sub.add_parser("benchmarks")
    p_bench.add_argument("--algo", choices=["ppo", "a2c", "sac", "td3"], default="ppo")
    p_bench.add_argument("--model", default=None)
    p_bench.set_defaults(func=cmd_benchmarks)

    p_diag = sub.add_parser("diagnostics")
    p_diag.add_argument("--algo", choices=["ppo", "a2c", "sac", "td3"], default="ppo")
    p_diag.add_argument("--model", default=None)
    p_diag.set_defaults(func=cmd_diagnostics)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
