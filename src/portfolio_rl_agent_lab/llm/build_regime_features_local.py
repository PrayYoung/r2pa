import os
import json
import numpy as np
import pandas as pd

from portfolio_rl_agent_lab.config import CFG
from portfolio_rl_agent_lab.data.io import load_news_features, load_returns
from portfolio_rl_agent_lab.features.market import compute_market_summary
from portfolio_rl_agent_lab.llm.store import save_regime_store
from portfolio_rl_agent_lab.llm.oracle_local import local_regime_from_summary
from portfolio_rl_agent_lab.text.news_loader import load_news_map
from portfolio_rl_agent_lab.text.encoder import NewsEncoder
from portfolio_rl_agent_lab.text.select_bullets import select_representative_bullets


def build_regime_features_local(
    returns_path: str = "artifacts/data/processed/returns.parquet",
    news_features_path: str = "artifacts/data/processed/news_features.parquet",
    cache_path: str = "artifacts/data/processed/regime_features_local_cache.jsonl",
    model: str = "llama3:latest",
    backend: str = "ollama",
    temperature: float = 0.0,
    max_steps: int = 5,
) -> pd.DataFrame:
    os.makedirs("artifacts/data/processed", exist_ok=True)

    rets = load_returns(returns_path)
    rets_np = rets.to_numpy(dtype=np.float32)
    ret_idx = pd.to_datetime(rets.index)

    news_df = load_news_features(news_features_path)

    news_map = load_news_map(ret_idx)
    encoder = NewsEncoder(model_name="all-MiniLM-L6-v2", device="cpu")


    # Cache to avoid re-querying the model
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                cache[obj["date"]] = obj["features"]

    rows, idx = [], []
    new_lines = []

    # for t in range(CFG.window, len(rets) - 2):
    max_t = min(len(rets) - 2, CFG.window + max_steps)
    for t in range(CFG.window, max_t):
        dt = pd.to_datetime(ret_idx[t])
        print(f"[build_local] date={dt.date()} start")
        dt_str = dt.strftime("%Y-%m-%d")

        window_rets = rets_np[t - CFG.window:t, :]
        summary = compute_market_summary(window_rets)

        # Merge Phase-1 text features into the summary dict
        if dt in news_df.index:
            nf = news_df.loc[dt]
            summary["news_count"] = float(nf.get("news_count", 0.0))
            summary["news_var"] = float(nf.get("news_var", 0.0))
            summary["news_shift"] = float(nf.get("news_shift", 0.0))
        else:
            summary["news_count"] = 0.0
            summary["news_var"] = 0.0
            summary["news_shift"] = 0.0

        if dt_str in cache:
            feat = cache[dt_str]
        else:
            texts = news_map.get(dt, [])
            print(f"[build_local] bullets={len(texts)}")
            emb = encoder.encode(texts)
            bullets = select_representative_bullets(texts, emb, k=3)
            print(f"[build_local] selected_k={len(bullets)}")
            rf = local_regime_from_summary(
                summary,
                news_bullets=bullets,
                backend=backend,
                model=model,
                temperature=temperature,
            )
            print(f"[build_local] got_llm_json")
            if t == CFG.window:
                print("Sample bullets:", bullets)

            feat = rf.to_vector()
            cache[dt_str] = feat
            new_lines.append({"date": dt_str, "summary": summary,
                              "bullets": bullets, "features": feat})

        rows.append(feat)
        idx.append(dt)

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
    print(f"Saved LOCAL vLLM regime features: {CFG.regime_store_path} | shape={df.shape}")
    return df

def main():
    build_regime_features_local()

if __name__ == "__main__":
    main()
