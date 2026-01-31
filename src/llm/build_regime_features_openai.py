import os
import json
import numpy as np
import pandas as pd

from src.config import CFG
from src.llm.store import save_regime_store
from src.llm.oracle_openai import openai_regime_from_summary

def load_returns(path="artifacts/data/processed/returns.parquet"):
    return pd.read_parquet(path)

def compute_market_summary(window_rets: np.ndarray) -> dict:
    mkt = window_rets.mean(axis=1)
    mkt_mom = float(np.prod(1.0 + mkt) - 1.0)
    mkt_vol = float(np.std(mkt) * np.sqrt(252.0))
    nav = np.cumprod(1.0 + mkt)
    peak = np.maximum.accumulate(nav)
    mkt_mdd = float((nav / peak - 1.0).min())
    return {"mkt_mom": mkt_mom, "mkt_vol": mkt_vol, "mkt_mdd": mkt_mdd}

def main():
    os.makedirs("artifacts/data/processed", exist_ok=True)
    rets = load_returns()
    rets_np = rets.to_numpy(dtype=np.float32)

    cache_path = "artifacts/data/processed/regime_features_openai_cache.jsonl"
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                cache[obj["date"]] = obj["features"]

    rows, idx = [], []
    new_lines = []

    for t in range(CFG.window, len(rets) - 2):
        dt = pd.to_datetime(rets.index[t]).strftime("%Y-%m-%d")
        window_rets = rets_np[t - CFG.window:t, :]
        summary = compute_market_summary(window_rets)

        if dt in cache:
            feat = cache[dt]
        else:
            rf = openai_regime_from_summary(summary, news_bullets=[])
            feat = rf.to_vector()
            cache[dt] = feat
            new_lines.append({"date": dt, "summary": summary, "features": feat})

        rows.append(feat)
        idx.append(pd.to_datetime(rets.index[t]))

    if new_lines:
        with open(cache_path, "a", encoding="utf-8") as f:
            for obj in new_lines:
                f.write(json.dumps(obj) + "\n")

    df = pd.DataFrame(
        rows,
        index=pd.to_datetime(idx),
        columns=["regime", "confidence", "macro_risk", "equity_bias", "defensive_bias"],
    )

    save_regime_store(df)
    print(f"Saved OpenAI regime features: {CFG.regime_store_path} | shape={df.shape}")

if __name__ == "__main__":
    main()
