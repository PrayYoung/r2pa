from __future__ import annotations

from portfolio_rl_agent_lab.train.train_rl import train_rl


def train_ppo(
    returns_path: str = "artifacts/data/processed/returns.parquet",
    model_path: str = "artifacts/models/ppo_portfolio",
    tb_log_path: str = "artifacts/tb_logs",
    total_timesteps: int = 200_000,
):
    return train_rl(
        algo="ppo",
        returns_path=returns_path,
        model_path=model_path,
        tb_log_path=tb_log_path,
        total_timesteps=total_timesteps,
    )


def main() -> None:
    train_ppo()


if __name__ == "__main__":
    main()
