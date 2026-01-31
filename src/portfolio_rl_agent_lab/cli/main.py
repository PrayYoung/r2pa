from __future__ import annotations

import argparse
import sys


def _dispatch(module_main, argv: list[str]) -> None:
    # Delegate subcommand args to the underlying module CLI.
    sys.argv = [sys.argv[0]] + argv
    module_main()


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio RL Agent Lab CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_data = sub.add_parser("data", help="Data utilities")
    p_data.add_argument("args", nargs=argparse.REMAINDER)

    p_rl = sub.add_parser("rl", help="RL training and evaluation")
    p_rl.add_argument("args", nargs=argparse.REMAINDER)

    p_student = sub.add_parser("student", help="Student model pipeline")
    p_student.add_argument("args", nargs=argparse.REMAINDER)

    p_teacher = sub.add_parser("teacher", help="Teacher/oracle feature builders")
    p_teacher.add_argument("args", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.cmd == "data":
        from portfolio_rl_agent_lab.cli.data import main as m
        _dispatch(m, args.args)
    elif args.cmd == "rl":
        from portfolio_rl_agent_lab.cli.rl import main as m
        _dispatch(m, args.args)
    elif args.cmd == "student":
        from portfolio_rl_agent_lab.cli.student import main as m
        _dispatch(m, args.args)
    elif args.cmd == "teacher":
        from portfolio_rl_agent_lab.cli.teacher import main as m
        _dispatch(m, args.args)
    else:
        parser.error(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
