from __future__ import annotations

import argparse


def cmd_build_dataset(_args):
    from src.student.build_student_dataset import main as m
    m()


def cmd_train(_args):
    from src.student.train_student_regime import main as m
    m()


def cmd_infer(_args):
    from src.student.build_regime_features_student import main as m
    m()


def main():
    parser = argparse.ArgumentParser(description="Student CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build-dataset").set_defaults(func=cmd_build_dataset)
    sub.add_parser("train").set_defaults(func=cmd_train)
    sub.add_parser("infer").set_defaults(func=cmd_infer)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
