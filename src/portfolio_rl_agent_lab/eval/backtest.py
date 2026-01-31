import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from portfolio_rl_agent_lab.config import CFG
from portfolio_rl_agent_lab.core.io import load_returns
from portfolio_rl_agent_lab.env.portfolio_env import PortfolioEnv

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

def run_backtest(
    returns_path: str = "artifacts/data/processed/returns.parquet",
    model_path: str = "artifacts/models/ppo_portfolio",
):
    returns = load_returns(returns_path)
    split = int(len(returns) * 0.8)

    test_returns = returns.iloc[split:].copy()

    model = PPO.load(model_path)

    nav, rets = backtest(model, test_returns)

    print("Test results:")
    print(f"Total return: {(nav[-1]-1)*100:.2f}%")
    print(f"Annualized Sharpe (rough): {np.mean(rets)/np.std(rets)*np.sqrt(252):.2f}")
    print(f"Max Drawdown: {((nav/np.maximum.accumulate(nav))-1).min()*100:.2f}%")
    return nav, rets

def main():
    run_backtest()

if __name__ == "__main__":
    main()
