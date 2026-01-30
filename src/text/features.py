import numpy as np
from typing import Dict

def summarize_embeddings(emb: np.ndarray, prev_emb: np.ndarray | None = None) -> Dict[str, float]:
    """
    Convert raw embeddings into low-dimensional text features.
    Aggressive compression 
    """
    if emb.size == 0:
        return {
            "news_count": 0.0,
            "news_var": 0.0,
            "news_shift": 0.0,
        }

    mean_emb = emb.mean(axis=0)
    var = float(np.mean(np.var(emb, axis=0)))

    if prev_emb is not None and prev_emb.size > 0:
        prev_mean = prev_emb.mean(axis=0)
        shift = float(1.0 - np.dot(mean_emb, prev_mean))
    else:
        shift = 0.0

    return {
        "news_count": float(len(emb)),
        "news_var": var,
        "news_shift": shift,
    }
