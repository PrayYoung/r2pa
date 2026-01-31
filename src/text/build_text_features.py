from __future__ import annotations

import os
import pandas as pd

from src.text.mock_news import load_mock_news
from src.text.encoder import NewsEncoder
from src.text.features import summarize_embeddings

def main():
    os.makedirs("artifacts/data/processed", exist_ok=True)

    rets = pd.read_parquet("artifacts/data/processed/returns.parquet")
    idx = pd.to_datetime(rets.index)

    news_map = load_mock_news(idx)
    encoder = NewsEncoder(model_name="all-MiniLM-L6-v2", device="cpu")

    rows = []
    prev_emb = None

    for dt in idx:
        texts = news_map.get(dt, [])
        emb = encoder.encode(texts)
        feats = summarize_embeddings(emb, prev_emb=prev_emb)
        rows.append(feats)
        prev_emb = emb

    df = pd.DataFrame(rows, index=idx).sort_index()
    out_path = "artifacts/data/processed/news_features.parquet"
    df.to_parquet(out_path)
    print(f"Saved text features: {out_path} | shape={df.shape}")

if __name__ == "__main__":
    main()
