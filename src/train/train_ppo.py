# src/train/train_ppo.py
import os
import torch
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.config import CFG
from src.env.portfolio_env import PortfolioEnv

def load_returns(path="data/processed/returns.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)

def main():
    os.makedirs("models", exist_ok=True)
    rets = load_returns()

    # 简单 time split：80% train
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
        tensorboard_log="tb_logs",
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.0,
    )

    model.learn(total_timesteps=200_000,
                tb_log_name="ppo_baseline")
    model.save("models/ppo_portfolio")
    print("Saved model to models/ppo_portfolio.zip")

if __name__ == "__main__":
    main()
