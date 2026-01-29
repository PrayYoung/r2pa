from dataclasses import dataclass
from typing import List

@dataclass
class RegimeFeatures:
    # regime: -1 risk_off, 0 neutral, 1 risk_on
    regime: int
    confidence: float
    macro_risk: float
    equity_bias: float
    defensive_bias: float

    def to_vector(self) -> List[float]:
        return [float(self.regime), float(self.confidence), float(self.macro_risk),
                float(self.equity_bias), float(self.defensive_bias)]
