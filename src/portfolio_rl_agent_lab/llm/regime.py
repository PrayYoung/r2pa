from dataclasses import dataclass
from typing import Dict

@dataclass
class RegimeFeatures:
    regime: int          # risk_off=-1, neutral=0, risk_on=1
    confidence: float    # 0~1
    macro_risk: float    # 0~1
    equity_bias: float   # 0~1
    defensive_bias: float# 0~1

    def to_vector(self):
        return [self.regime, self.confidence, self.macro_risk, self.equity_bias, self.defensive_bias]

def heuristic_regime(mkt_summary: Dict) -> RegimeFeatures:
    """
    """
    # placeholder
    vol = float(mkt_summary.get("vol_20d", 0.2))
    mom = float(mkt_summary.get("mom_60d", 0.0))
    risk_on = 1 if (mom > 0 and vol < 0.25) else (-1 if (mom < 0 and vol > 0.25) else 0)
    conf = 0.6
    return RegimeFeatures(
        regime=risk_on,
        confidence=conf,
        macro_risk=min(1.0, max(0.0, (vol - 0.15) / 0.25)),
        equity_bias=0.6 if risk_on == 1 else 0.4,
        defensive_bias=0.6 if risk_on == -1 else 0.4,
    )
