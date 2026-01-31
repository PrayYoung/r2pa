from __future__ import annotations

import os
import numpy as np
import pandas as pd

from portfolio_rl_agent_lab.config import CFG


def compute_market_summary(window_rets: np.ndarray) -> dict:
    mkt = window_rets.mean(axis=1)
    mkt_mom = float(np.prod(1.0 + mkt) - 1.0)
    mkt_vol = float(np.std(mkt) * np.sqrt(252.0))
    nav = np.cumprod(1.0 + mkt)
    peak = np.maximum.accumulate(nav)
    mkt_mdd = float((nav / peak - 1.0).min())
    return {"mkt_mom": mkt_mom, "mkt_vol": mkt_vol, "mkt_mdd": mkt_mdd}


def build_student_dataset(
    returns_path: str = "artifacts/data/processed/returns.parquet",
    news_features_path: str = "artifacts/data/processed/news_features.parquet",
    out_path: str = "artifacts/data/processed/student_dataset.parquet",
) -> pd.DataFrame:
    os.makedirs("artifacts/data/processed", exist_ok=True)

    rets = pd.read_parquet(returns_path)
    rets_np = rets.to_numpy(dtype=np.float32)
    idx = pd.to_datetime(rets.index)

    # Teacher labels: produced by LLM oracle build (local)
    teacher = pd.read_parquet(CFG.regime_store_path)
    teacher.index = pd.to_datetime(teacher.index)
    teacher = teacher.sort_index()

    # Numeric news features (3-dim)
    news = pd.read_parquet(news_features_path)
    news.index = pd.to_datetime(news.index)
    news = news.sort_index()

    rows = []
    for t in range(CFG.window, len(rets) - 2):
        dt = idx[t]
        if dt not in teacher.index:
            continue

        window_rets = rets_np[t - CFG.window:t, :]
        ms = compute_market_summary(window_rets)

        if dt in news.index:
            nf = news.loc[dt]
            news_count = float(nf.get("news_count", 0.0))
            news_var = float(nf.get("news_var", 0.0))
            news_shift = float(nf.get("news_shift", 0.0))
        else:
            news_count, news_var, news_shift = 0.0, 0.0, 0.0

        y = teacher.loc[dt][
            ["regime", "confidence", "macro_risk", "equity_bias", "defensive_bias"]
        ].to_numpy(dtype=np.float32)

        rows.append(
            {
                "date": dt,
                "mkt_mom": ms["mkt_mom"],
                "mkt_vol": ms["mkt_vol"],
                "mkt_mdd": ms["mkt_mdd"],
                "news_count": news_count,
                "news_var": news_var,
                "news_shift": news_shift,
                "y_regime": float(y[0]),
                "y_conf": float(y[1]),
                "y_macro_risk": float(y[2]),
                "y_equity_bias": float(y[3]),
                "y_defensive_bias": float(y[4]),
            }
        )

    df = pd.DataFrame(rows).set_index("date").sort_index()
    df.to_parquet(out_path)
    print(f"Saved: {out_path} | shape={df.shape} | {df.index.min()}..{df.index.max()}")
    return df

def main():
    build_student_dataset()

if __name__ == "__main__":
    main()
