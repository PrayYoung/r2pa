from __future__ import annotations

from typing import Dict, List
import json
import os
import pandas as pd

from src.text.mock_news import load_mock_news


def load_news_map_from_jsonl(path: str, index: pd.DatetimeIndex) -> Dict[pd.Timestamp, List[str]]:
    """
    Load daily bullets from JSONL and align them to a given DatetimeIndex.
    Missing dates will map to empty list.
    """
    m: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            m[obj["date"]] = obj.get("bullets", [])

    out: Dict[pd.Timestamp, List[str]] = {}
    for dt in index:
        key = pd.to_datetime(dt).strftime("%Y-%m-%d")
        out[pd.to_datetime(dt)] = m.get(key, [])
    return out


def load_news_map(index: pd.DatetimeIndex, path: str = "artifacts/data/processed/news_daily.jsonl"):
    """
    Unified entry: real Alpaca news if available; otherwise fall back to mock.
    """
    if os.path.exists(path):
        return load_news_map_from_jsonl(path, index=index)
    return load_mock_news(index)
