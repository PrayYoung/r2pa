from __future__ import annotations

import numpy as np


def compute_market_summary(window_rets: np.ndarray) -> dict:
    """
    window_rets: (window, N_assets) returns window ending at t (exclusive of t+1)
    """
    mkt = window_rets.mean(axis=1)
    mkt_mom = float(np.prod(1.0 + mkt) - 1.0)
    mkt_vol = float(np.std(mkt) * np.sqrt(252.0))
    nav = np.cumprod(1.0 + mkt)
    peak = np.maximum.accumulate(nav)
    mkt_mdd = float((nav / peak - 1.0).min())
    return {"mkt_mom": mkt_mom, "mkt_vol": mkt_vol, "mkt_mdd": mkt_mdd}
