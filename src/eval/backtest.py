import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from src.config import CFG
from src.env.portfolio_env import PortfolioEnv

def load_returns(path="artifacts/data/processed/returns.parquet"):
    return pd.read_parquet(path)

def backtest(model, returns: pd.DataFrame):
    env = PortfolioEnv(
        returns,
        window=CFG.window,
        trading_cost_bps=CFG.trading_cost_bps,
        cash_asset=CFG.cash_asset,
    )

    obs, _ = env.reset()
    done = False

    nav = [1.0]
    rets = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        nav.append(nav[-1] * (1.0 + info["port_ret"] - info["cost"]))
        rets.append(info["port_ret"] - info["cost"])

        if truncated or terminated:
            break

    nav = np.array(nav)
    rets = np.array(rets)
    return nav, rets

def main():
    returns = load_returns()
    split = int(len(returns) * 0.8)

    test_returns = returns.iloc[split:].copy()

    model = PPO.load("artifacts/models/ppo_portfolio")

    nav, rets = backtest(model, test_returns)

    print("Test results:")
    print(f"Total return: {(nav[-1]-1)*100:.2f}%")
    print(f"Annualized Sharpe (rough): {np.mean(rets)/np.std(rets)*np.sqrt(252):.2f}")
    print(f"Max Drawdown: {((nav/np.maximum.accumulate(nav))-1).min()*100:.2f}%")

if __name__ == "__main__":
    main()
