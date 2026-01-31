import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from portfolio_rl_agent_lab.config import CFG
from portfolio_rl_agent_lab.core.io import load_returns
from portfolio_rl_agent_lab.env.portfolio_env import PortfolioEnv

def run_diagnostics(
    returns_path: str = "artifacts/data/processed/returns.parquet",
    model_path: str = "artifacts/models/ppo_portfolio",
):
    returns = load_returns(returns_path)
    split = int(len(returns) * 0.8)
    test_returns = returns.iloc[split:].copy()

    model = PPO.load(model_path)

    env = PortfolioEnv(test_returns, window=CFG.window, trading_cost_bps=CFG.trading_cost_bps, cash_asset=CFG.cash_asset)
    obs, _ = env.reset()

    weights = []
    turnovers = []
    costs = []
    port_rets = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        turnovers.append(info["turnover"])
        costs.append(info["cost"])
        port_rets.append(info["port_ret"])
        a = np.clip(action.astype(np.float32), 0.0, None)
        s = float(a.sum())
        if s < 1e-8:
            a = np.ones_like(a) / len(a)
        else:
            a = a / s
        weights.append(a)

        if truncated or terminated:
            break

    W = np.vstack(weights)  # (T, N_action)
    avg_w = W.mean(axis=0)
    std_w = W.std(axis=0)

    asset_names = list(test_returns.columns)
    if CFG.cash_asset:
        asset_names = asset_names + ["CASH"]

    print("\n=== WEIGHT DIAGNOSTICS (Test) ===")
    top = np.argsort(-avg_w)
    for i in top[: min(10, len(top))]:
        print(f"{asset_names[i]:>6}: avg={avg_w[i]:.3f}  std={std_w[i]:.3f}")

    turnovers = np.array(turnovers)
    costs = np.array(costs)
    port_rets = np.array(port_rets)

    print("\n=== TURNOVER / COST ===")
    print(f"Avg turnover (L1): {turnovers.mean():.3f}")
    print(f"Median turnover:   {np.median(turnovers):.3f}")
    print(f"Avg daily cost:    {costs.mean()*100:.3f}%")
    print(f"Avg daily ret:     {port_rets.mean()*100:.3f}%")
    return {
        "avg_w": avg_w,
        "std_w": std_w,
        "turnovers": turnovers,
        "costs": costs,
        "port_rets": port_rets,
        "asset_names": asset_names,
    }

def main():
    run_diagnostics()

if __name__ == "__main__":
    main()
