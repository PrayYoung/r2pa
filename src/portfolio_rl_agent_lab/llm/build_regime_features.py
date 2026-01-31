import os
import numpy as np
import pandas as pd

from portfolio_rl_agent_lab.config import CFG
from portfolio_rl_agent_lab.llm.oracle_heuristic import heuristic_from_summary
from portfolio_rl_agent_lab.llm.store import save_regime_store


def load_returns(path: str = "artifacts/data/processed/returns.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


def load_news_features(path: str = "artifacts/data/processed/news_features.parquet") -> pd.DataFrame:
    """
    Load daily text-derived features:
      - news_count
      - news_var
      - news_shift
    """
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def compute_market_summary(window_rets: np.ndarray) -> dict:
    """
    window_rets: (window, N_assets) returns window ending at t (exclusive of t+1)
    """
    # Equal-weight market proxy
    mkt = window_rets.mean(axis=1)  # (window,)
    mkt_mom = float(np.prod(1.0 + mkt) - 1.0)         # cumulative over window
    mkt_vol = float(np.std(mkt) * np.sqrt(252.0))     # annualized volatility

    # Max drawdown on market NAV within the window
    nav = np.cumprod(1.0 + mkt)
    peak = np.maximum.accumulate(nav)
    mkt_mdd = float((nav / peak - 1.0).min())

    return {"mkt_mom": mkt_mom, "mkt_vol": mkt_vol, "mkt_mdd": mkt_mdd}


def build_regime_features_heuristic(
    returns_path: str = "artifacts/data/processed/returns.parquet",
    news_features_path: str = "artifacts/data/processed/news_features.parquet",
) -> pd.DataFrame:
    os.makedirs("artifacts/data/processed", exist_ok=True)

    rets = load_returns(returns_path)
    rets_np = rets.to_numpy(dtype=np.float32)

    # Load text-derived features (Phase 1: 3-dim compressed features)
    news_df = load_news_features(news_features_path)
    ret_idx = pd.to_datetime(rets.index)

    rows = []
    idx = []

    # Generate features aligned with env time index t (the "current" time with history available)
    # Env uses window history up to t, and applies returns at t+1.
    for t in range(CFG.window, len(rets) - 2):
        dt = pd.to_datetime(ret_idx[t])

        window_rets = rets_np[t - CFG.window:t, :]  # (window, N)
        summary = compute_market_summary(window_rets)

        # Merge text features into the same summary dict
        if dt in news_df.index:
            nf = news_df.loc[dt]
            summary["news_count"] = float(nf.get("news_count", 0.0))
            summary["news_var"] = float(nf.get("news_var", 0.0))
            summary["news_shift"] = float(nf.get("news_shift", 0.0))
        else:
            summary["news_count"] = 0.0
            summary["news_var"] = 0.0
            summary["news_shift"] = 0.0

        feat = heuristic_from_summary(summary).to_vector()
        rows.append(feat)
        idx.append(dt)  # feature timestamp at "t"

    df = pd.DataFrame(
        rows,
        index=pd.to_datetime(idx),
        columns=["regime", "confidence", "macro_risk", "equity_bias", "defensive_bias"],
    )

    save_regime_store(df, path=CFG.regime_store_heuristic_path)
    print(
        f"Saved regime features: {CFG.regime_store_heuristic_path} | "
        f"shape={df.shape} | {df.index.min()}..{df.index.max()}"
    )
    return df


def main():
    build_regime_features_heuristic()


if __name__ == "__main__":
    main()
