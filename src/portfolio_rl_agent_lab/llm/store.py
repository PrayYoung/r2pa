import pandas as pd
from portfolio_rl_agent_lab.config import CFG

def load_regime_store() -> pd.DataFrame:
    """
    Load regime features store based on CFG.regime_source.
    Returned df index must be DatetimeIndex.
    """
    src = getattr(CFG, "regime_source", "llm")

    if src == "student":
        path = CFG.regime_store_student_path
    elif src == "heuristic":
        path = CFG.regime_store_heuristic_path
    else:
        path = CFG.regime_store_path

    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df

def save_regime_store(df: pd.DataFrame, path: str = None):
    path = path or CFG.regime_store_path
    df.sort_index().to_parquet(path)
