import numpy as np
from src.llm.schema import RegimeFeatures

def heuristic_from_summary(summary: dict) -> RegimeFeatures:
    """
    summary from numbers only:momentum/vol/drawdown/corr/breadth...
    minimum version :mom + vol decide risk-on/off。
    """
    mom = float(summary.get("mkt_mom", 0.0))   # e.g. 60d cumulative return
    vol = float(summary.get("mkt_vol", 0.2))   # e.g. annualized vol

    # momentum > 0 and vol low -> risk_on；
    # momentum < 0 and vol high -> risk_off
    # otherwise neutral
    if mom > 0.0 and vol < 0.25:
        regime = 1
        conf = 0.70
    elif mom < 0.0 and vol > 0.25:
        regime = -1
        conf = 0.70
    else:
        regime = 0
        conf = 0.55

    macro_risk = float(np.clip((vol - 0.15) / 0.25, 0.0, 1.0))
    equity_bias = 0.70 if regime == 1 else (0.45 if regime == 0 else 0.25)
    defensive_bias = 1.0 - equity_bias

    return RegimeFeatures(
        regime=regime,
        confidence=conf,
        macro_risk=macro_risk,
        equity_bias=float(equity_bias),
        defensive_bias=float(defensive_bias),
    )
