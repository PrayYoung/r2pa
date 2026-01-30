import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from src.config import CFG
from src.env.portfolio_env import PortfolioEnv

def load_returns(path="data/processed/returns.parquet"):
    return pd.read_parquet(path)

def max_drawdown(nav: np.ndarray) -> float:
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    return float(dd.min())

def sharpe(rets: np.ndarray, ann=252) -> float:
    std = np.std(rets)
    if std < 1e-12:
        return 0.0
    return float(np.mean(rets) / std * np.sqrt(ann))

def run_rl_policy(model: PPO, returns: pd.DataFrame):
    env = PortfolioEnv(returns, window=CFG.window, trading_cost_bps=CFG.trading_cost_bps, cash_asset=CFG.cash_asset)
    obs, _ = env.reset()

    nav = [1.0]
    rets = []
    dates = returns.index[CFG.window+1:]  # roughly align with stepping

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        step_ret = info["port_ret"] - info["cost"]
        nav.append(nav[-1] * (1.0 + step_ret))
        rets.append(step_ret)
        if truncated or terminated:
            break

    return np.array(nav), np.array(rets), dates[: len(rets)]

def run_equal_weight_daily(returns: pd.DataFrame):
    n = returns.shape[1]
    n_action = n + (1 if CFG.cash_asset else 0)
    w = np.zeros(n_action, dtype=np.float32)
    # equal weight across assets, cash=0
    w[:n] = 1.0 / n
    if CFG.cash_asset:
        w[-1] = 0.0

    nav = [1.0]
    rets = []
    dates = returns.index[CFG.window+1:]

    # daily rebalance to equal weight => turnover each day depends on drift, but we simulate with same env for fairness
    env = PortfolioEnv(returns, window=CFG.window, trading_cost_bps=CFG.trading_cost_bps, cash_asset=CFG.cash_asset)
    obs, _ = env.reset()
    while True:
        obs, reward, terminated, truncated, info = env.step(w)
        step_ret = info["port_ret"] - info["cost"]
        nav.append(nav[-1] * (1.0 + step_ret))
        rets.append(step_ret)
        if truncated or terminated:
            break

    return np.array(nav), np.array(rets), dates[: len(rets)]

def run_buy_and_hold_equal_weight(returns: pd.DataFrame):
    n = returns.shape[1]
    n_action = n + (1 if CFG.cash_asset else 0)

    w0 = np.zeros(n_action, dtype=np.float32)
    w0[:n] = 1.0 / n
    if CFG.cash_asset:
        w0[-1] = 0.0

    # one-time initial turnover cost from uniform start (env starts equal-weight incl cash)
    # we'll compute it directly for simplicity
    w_start = np.ones(n_action, dtype=np.float32) / n_action
    turnover0 = float(np.abs(w0 - w_start).sum())
    cost0 = turnover0 * (CFG.trading_cost_bps / 10000.0)

    nav = [1.0 * (1.0 - cost0)]
    rets = []
    dates = returns.index[CFG.window+1:]

    # Buy&hold: no further turnover/cost; weights drift implicitly, but for simplicity we assume constant weights on returns.
    # More exact: simulate weight drift; we keep it simple baseline here.
    # Use constant weights on next-day returns:
    rets_mat = returns.to_numpy(dtype=np.float32)
    t_min = CFG.window
    t_max = len(returns) - 2
    for t in range(t_min, t_max + 1):
        r_next_assets = rets_mat[t + 1, :]
        if CFG.cash_asset:
            r_vec = np.concatenate([r_next_assets, np.array([0.0], dtype=np.float32)])
        else:
            r_vec = r_next_assets
        step_ret = float((w0 * r_vec).sum())
        nav.append(nav[-1] * (1.0 + step_ret))
        rets.append(step_ret)

    return np.array(nav), np.array(rets), dates[: len(rets)]

def summarize(name: str, nav: np.ndarray, rets: np.ndarray):
    total = (nav[-1] - 1.0) * 100
    s = sharpe(rets)
    dd = max_drawdown(nav) * 100
    print(f"{name:>18} | Total {total:>7.2f}% | Sharpe {s:>5.2f} | MaxDD {dd:>7.2f}%")

def main():
    returns = load_returns()
    split = int(len(returns) * 0.8)
    test_returns = returns.iloc[split:].copy()

    model = PPO.load("models/ppo_portfolio")

    nav_rl, rets_rl, dates = run_rl_policy(model, test_returns)
    nav_eq, rets_eq, _ = run_equal_weight_daily(test_returns)
    nav_bh, rets_bh, _ = run_buy_and_hold_equal_weight(test_returns)

    print("\n=== TEST BENCHMARKS (out-of-sample) ===")
    summarize("RL PPO", nav_rl, rets_rl)
    summarize("EqualW Daily", nav_eq, rets_eq)
    summarize("BuyHold EqualW", nav_bh, rets_bh)

    # align lengths for plotting
    y1 = nav_rl[1:]
    y2 = nav_eq[1:]
    y3 = nav_bh[1:]
    L = min(len(dates), len(y1), len(y2), len(y3))


    # Plot
    plt.figure()
    plt.plot(dates[:L], y1[:L], label="RL PPO")
    plt.plot(dates[:L], y2[:L], label="EqualW Daily")
    plt.plot(dates[:L], y3[:L], label="BuyHold EqualW")
    plt.title("Out-of-sample NAV (Test)")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/processed/test_nav.png", dpi=160)
    print("\nSaved plot: data/processed/test_nav.png")

if __name__ == "__main__":
    main()
