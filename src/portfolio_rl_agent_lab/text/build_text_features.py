from __future__ import annotations

import os
import pandas as pd

from portfolio_rl_agent_lab.text.mock_news import load_mock_news
from portfolio_rl_agent_lab.text.encoder import NewsEncoder
from portfolio_rl_agent_lab.text.features import summarize_embeddings

def build_text_features(
    returns_path: str = "artifacts/data/processed/returns.parquet",
    out_path: str = "artifacts/data/processed/news_features.parquet",
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
) -> pd.DataFrame:
    os.makedirs("artifacts/data/processed", exist_ok=True)

    rets = pd.read_parquet(returns_path)
    idx = pd.to_datetime(rets.index)

    news_map = load_mock_news(idx)
    encoder = NewsEncoder(model_name=model_name, device=device)

    rows = []
    prev_emb = None

    for dt in idx:
        texts = news_map.get(dt, [])
        emb = encoder.encode(texts)
        feats = summarize_embeddings(emb, prev_emb=prev_emb)
        rows.append(feats)
        prev_emb = emb

    df = pd.DataFrame(rows, index=idx).sort_index()
    df.to_parquet(out_path)
    print(f"Saved text features: {out_path} | shape={df.shape}")
    return df

def main():
    build_text_features()

if __name__ == "__main__":
    main()
