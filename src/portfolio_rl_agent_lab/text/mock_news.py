from __future__ import annotations

from typing import Dict, List
import pandas as pd

def load_mock_news(index: pd.DatetimeIndex) -> Dict[pd.Timestamp, List[str]]:
    """
    Return a mapping: date -> list of news strings.
    This is a placeholder to validate the pipeline end-to-end.
    Replace it later with a real news loader.
    """
    news = {}

    for dt in index:
        # Default: no news
        news[dt] = []

    # Add a few synthetic examples (edit freely)
    if len(index) > 50:
        news[index[20]] = [
            "Fed officials signal rates may stay higher for longer.",
            "Equity volatility rises as macro uncertainty increases."
        ]
        news[index[35]] = [
            "AI-related stocks rally on strong demand outlook.",
            "Semiconductor supply chain shows signs of improvement."
        ]
        news[index[45]] = [
            "Oil prices jump amid rising geopolitical tensions.",
            "Investors rotate toward defensives and gold."
        ]

    return news
