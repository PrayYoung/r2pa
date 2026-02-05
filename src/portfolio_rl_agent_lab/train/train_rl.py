from __future__ import annotations

import argparse
import os

import torch
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

from portfolio_rl_agent_lab.config import CFG
from portfolio_rl_agent_lab.data.io import load_returns
from portfolio_rl_agent_lab.rl.registry import normalize_algo
from portfolio_rl_agent_lab.env.portfolio_env import PortfolioEnv


def train_rl(
    algo: str = "ppo",
    returns_path: str = "artifacts/data/processed/returns.parquet",
    model_path: str | None = None,
    tb_log_path: str = "artifacts/tb_logs",
    total_timesteps: int = 200_000,
):
    algo = normalize_algo(algo)
    os.makedirs("artifacts/models", exist_ok=True)

    rets = load_returns(returns_path)
    split = int(len(rets) * 0.8)
    train_rets = rets.iloc[:split].copy()

    def make_env():
        return PortfolioEnv(
            train_rets,
            window=CFG.window,
            trading_cost_bps=CFG.trading_cost_bps,
            cash_asset=CFG.cash_asset,
        )

    env = DummyVecEnv([make_env])
    model_path = model_path or f"artifacts/models/{algo}_portfolio"

    if algo in {"ppo", "a2c"}:
        policy_kwargs = {
            "net_arch": {"pi": [128, 128], "vf": [128, 128]},
            "activation_fn": torch.nn.ReLU,
        }
    else:
        policy_kwargs = {"net_arch": [256, 256], "activation_fn": torch.nn.ReLU}

    if algo == "ppo":
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_log_path,
            n_steps=2048,
            batch_size=256,
            gamma=0.99,
            learning_rate=3e-4,
            ent_coef=0.0,
        )
    elif algo == "a2c":
        model = A2C(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_log_path,
            n_steps=5,
            gamma=0.99,
            learning_rate=7e-4,
        )
    elif algo == "sac":
        model = SAC(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_log_path,
            buffer_size=100_000,
            learning_rate=3e-4,
            batch_size=256,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
        )
    else:
        action_dim = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(action_dim), sigma=0.1 * np.ones(action_dim))
        model = TD3(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_log_path,
            action_noise=action_noise,
            buffer_size=100_000,
            learning_rate=3e-4,
            batch_size=256,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
        )

    model.learn(total_timesteps=total_timesteps, tb_log_name=f"{algo}_baseline")
    model.save(model_path)
    print(f"Saved model to {model_path}.zip")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL agent")
    parser.add_argument("--algo", choices=["ppo", "a2c", "sac", "td3"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--returns", default="artifacts/data/processed/returns.parquet")
    args = parser.parse_args()

    train_rl(
        algo=args.algo,
        returns_path=args.returns,
        model_path=args.model_path,
        total_timesteps=args.timesteps,
    )


if __name__ == "__main__":
    main()
