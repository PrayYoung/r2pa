# src/env/portfolio_env.py
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from src.config import CFG

def _simplex_project(w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # non-negative + sum to 1
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if s < eps:
        w = np.ones_like(w) / len(w)
    else:
        w = w / s
    return w

class PortfolioEnv(gym.Env):
    """
    Observation: past window returns (N x window flattened) [+ current weights]
    Action: target weights for N (+ cash if enabled)
    Reward: next-day portfolio return - turnover_cost
    """
    metadata = {"render_modes": []}

    def __init__(self, returns: pd.DataFrame, window: int = None, trading_cost_bps: float = None, cash_asset: bool = None):
        super().__init__()
        self.returns = returns.copy()
        self.window = window if window is not None else CFG.window
        self.trading_cost_bps = trading_cost_bps if trading_cost_bps is not None else CFG.trading_cost_bps
        self.cash_asset = cash_asset if cash_asset is not None else CFG.cash_asset

        self.assets = list(self.returns.columns)
        self.n_assets = len(self.assets)
        self.n_action = self.n_assets + (1 if self.cash_asset else 0)  # cash as last dim

        # Regime / LLM features
        self.use_regime = CFG.use_regime_features
        self.regime_dim = CFG.regime_dim
        self._regime_df = None
        if self.use_regime:
            from src.llm.store import load_regime_store
            self._regime_df = load_regime_store()

            # Strict alignment: only carry forward past info; never backfill from the future.
            self._regime_df = self._regime_df.sort_index()
            ret_idx = pd.to_datetime(self.returns.index)
            self._regime_df = self._regime_df.reindex(ret_idx).ffill()
            # If the very beginning has no previous value to forward-fill, fill with zeros (no signal).
            self._regime_df = self._regime_df.fillna(0.0)

        # Action space: continuous weights (we will project to simplex)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_action,), dtype=np.float32)

        # Observation space: returns window flattened + current weights
        obs_dim = self.n_assets * self.window + self.n_action + (self.regime_dim if self.use_regime else 0)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)

        self._t = None
        self._w = None  # current weights (including cash if enabled)

        # Preconvert returns to numpy for speed
        self._rets = self.returns.to_numpy(dtype=np.float32)

        # t ranges so that we have window history and one-step-ahead return
        self._t_min = self.window
        self._t_max = len(self.returns) - 2  # need t+1
        if self._t_max <= self._t_min:
            raise ValueError("Not enough data for given window.")


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t = self._t_min
        self._w = np.ones(self.n_action, dtype=np.float32) / self.n_action  # start equal-weight incl cash
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        window_rets = self._rets[self._t - self.window:self._t, :]  # (window, N)
        feat = window_rets.T.reshape(-1)  # (N*window,)
        parts = [feat, self._w]

        if self.use_regime:
            # align by current date index
            dt = pd.to_datetime(self.returns.index[self._t])
            row = self._regime_df.loc[dt].to_numpy(dtype=np.float32)
            if row.shape[0] != self.regime_dim:
                raise ValueError(f"Regime dim mismatch: got {row.shape[0]}, expected {self.regime_dim}")
            parts.append(row)

        obs = np.concatenate(parts, axis=0).astype(np.float32)
        return obs

    def step(self, action):
        # 1) map action -> target weights on simplex
        a = np.array(action, dtype=np.float32)
        w_target = _simplex_project(a)

        # 2) compute turnover & cost
        turnover = float(np.abs(w_target - self._w).sum())
        cost = turnover * (self.trading_cost_bps / 10000.0)  # bps -> fraction

        # 3) realize next-day portfolio return
        r_next_assets = self._rets[self._t + 1, :]  # shape (N,)
        if self.cash_asset:
            r_vec = np.concatenate([r_next_assets, np.array([0.0], dtype=np.float32)])  # cash return ~ 0
        else:
            r_vec = r_next_assets

        port_ret = float((w_target * r_vec).sum())

        # 4) reward
        # reward = port_ret - cost                  # basic reward   
        downside_pen = CFG.downside_lambda * max(0.0, -port_ret)
        turnover_pen = CFG.turnover_lambda * turnover
        reward = port_ret - cost - downside_pen - turnover_pen

        # advance time & set weights
        self._w = w_target
        self._t += 1

        terminated = False
        truncated = self._t >= self._t_max
        info = {"port_ret": port_ret, "cost": cost, "turnover": turnover}

        return self._get_obs(), reward, terminated, truncated, info
