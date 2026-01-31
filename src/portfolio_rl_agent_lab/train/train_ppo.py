# src/train/train_ppo.py
import os
import torch
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from portfolio_rl_agent_lab.config import CFG
from portfolio_rl_agent_lab.core.io import load_returns
from portfolio_rl_agent_lab.env.portfolio_env import PortfolioEnv

def train_ppo(
    returns_path: str = "artifacts/data/processed/returns.parquet",
    model_path: str = "artifacts/models/ppo_portfolio",
    tb_log_path: str = "artifacts/tb_logs",
    total_timesteps: int = 200_000,
) -> PPO:
    os.makedirs("artifacts/models", exist_ok=True)
    rets = load_returns(returns_path)

    # 80% train
    split = int(len(rets) * 0.8)
    train_rets = rets.iloc[:split].copy()

    def make_env():
        return PortfolioEnv(train_rets, window=CFG.window, trading_cost_bps=CFG.trading_cost_bps, cash_asset=CFG.cash_asset)

    env = DummyVecEnv([make_env])

    policy_kwargs = dict(
        net_arch=dict(
            pi=[128, 128],
            vf=[128, 128],
        ),
        activation_fn=torch.nn.ReLU,
    )

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

    model.learn(total_timesteps=total_timesteps,
                tb_log_name="ppo_baseline")
    model.save(model_path)
    print(f"Saved model to {model_path}.zip")
    return model

def main():
    train_ppo()

if __name__ == "__main__":
    main()
