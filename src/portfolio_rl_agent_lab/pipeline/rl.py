from __future__ import annotations


def run_rl_pipeline(algo: str = "ppo") -> None:
    from portfolio_rl_agent_lab.train.train_rl import train_rl
    from portfolio_rl_agent_lab.eval.backtest import run_backtest
    from portfolio_rl_agent_lab.eval.benchmarks import run_benchmarks
    from portfolio_rl_agent_lab.eval.diagnostics import run_diagnostics

    model_path = f"artifacts/models/{algo}_portfolio"
    train_rl(algo=algo, model_path=model_path)
    run_benchmarks(algo=algo, model_path=model_path)
    run_diagnostics(algo=algo, model_path=model_path)
    run_backtest(algo=algo, model_path=model_path)


if __name__ == "__main__":
    run_rl_pipeline()
