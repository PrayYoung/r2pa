import os
import numpy as np
import pandas as pd
from src.config import CFG
from src.llm.oracle_heuristic import heuristic_from_summary
from src.llm.store import save_regime_store

def load_returns(path="data/processed/returns.parquet"):
    return pd.read_parquet(path)

def compute_market_summary(window_rets: np.ndarray) -> dict:
    """
    window_rets: (window, N_assets) returns window ending at t (exclusive of t+1)
    """
    # equal-weight market proxy
    mkt = window_rets.mean(axis=1)  # (window,)
    mkt_mom = float(np.prod(1.0 + mkt) - 1.0)         # cumulative over window
    mkt_vol = float(np.std(mkt) * np.sqrt(252.0))     # annualized
    # max drawdown on mkt nav within window
    nav = np.cumprod(1.0 + mkt)
    peak = np.maximum.accumulate(nav)
    mkt_mdd = float((nav / peak - 1.0).min())
    return {"mkt_mom": mkt_mom, "mkt_vol": mkt_vol, "mkt_mdd": mkt_mdd}

def main():
    os.makedirs("data/processed", exist_ok=True)
    rets = load_returns()
    rets_np = rets.to_numpy(dtype=np.float32)

    rows = []
    idx = []

    # We will generate features aligned with env time index t (the "current" time with history available)
    # Env uses window history up to t, and applies returns at t+1.
    for t in range(CFG.window, len(rets) - 2):
        window_rets = rets_np[t - CFG.window:t, :]  # (window, N)
        summary = compute_market_summary(window_rets)
        feat = heuristic_from_summary(summary).to_vector()
        rows.append(feat)
        idx.append(rets.index[t])  # features timestamp at "t"

    df = pd.DataFrame(
        rows,
        index=pd.to_datetime(idx),
        columns=["regime", "confidence", "macro_risk", "equity_bias", "defensive_bias"],
    )

    save_regime_store(df)
    print(f"Saved regime features: {CFG.regime_store_path} | shape={df.shape} | {df.index.min()}..{df.index.max()}")

if __name__ == "__main__":
    main()
