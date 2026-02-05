from __future__ import annotations

import argparse


def cmd_data(_args):
    from portfolio_rl_agent_lab.pipeline.data import run_data_pipeline
    run_data_pipeline()


def cmd_text(_args):
    from portfolio_rl_agent_lab.pipeline.text import run_text_pipeline
    run_text_pipeline()


def cmd_regime(args):
    from portfolio_rl_agent_lab.pipeline.regime import run_regime_pipeline
    run_regime_pipeline(args.source)


def cmd_student(_args):
    from portfolio_rl_agent_lab.pipeline.student import run_student_pipeline
    run_student_pipeline()


def cmd_rl(args):
    from portfolio_rl_agent_lab.pipeline.rl import run_rl_pipeline
    run_rl_pipeline(algo=args.algo)


def cmd_all(args):
    from portfolio_rl_agent_lab.pipeline.all import run_all_pipeline
    run_all_pipeline(args.source, algo=args.algo)


def main():
    parser = argparse.ArgumentParser(description="Pipeline CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("data").set_defaults(func=cmd_data)
    sub.add_parser("text").set_defaults(func=cmd_text)

    p_regime = sub.add_parser("regime")
    p_regime.add_argument("--source", choices=["heuristic", "local", "student"], default="heuristic")
    p_regime.set_defaults(func=cmd_regime)

    sub.add_parser("student").set_defaults(func=cmd_student)
    p_rl = sub.add_parser("rl")
    p_rl.add_argument("--algo", choices=["ppo", "a2c", "sac", "td3"], default="ppo")
    p_rl.set_defaults(func=cmd_rl)

    p_all = sub.add_parser("all")
    p_all.add_argument("--source", choices=["heuristic", "local", "student"], default="heuristic")
    p_all.add_argument("--algo", choices=["ppo", "a2c", "sac", "td3"], default="ppo")
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
