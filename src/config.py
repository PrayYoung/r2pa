from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    tickers: List[str] = None
    start: str = "2015-01-01"
    end: str = "2025-12-31"
    window: int = 60               # past 60 days
    rebalance_every: int = 1       # daily portfolio rebalancing
    trading_cost_bps: float = 10.0 # cost：10 bps（0.10%）
    risk_aversion: float = 0.0
    cash_asset: bool = True        # include cash as an asset
    downside_lambda: float = 0.2   # for downside risk measure
    turnover_lambda: float = 0.1
    use_regime_features: bool = True   # default turned off
    regime_dim: int = 5                 # C1 features: [regime, conf, macro_risk, equity_bias, defensive_bias]
    regime_source = "student"         # Regime feature source: "heuristic" | "llm" | "student"
    regime_store_path = "artifacts/data/processed/regime_features.parquet"
    regime_store_student_path = "artifacts/data/processed/regime_features_student.parquet"
    regime_store_heuristic_path = "artifacts/data/processed/regime_features_heuristic.parquet"


    def __post_init__(self):
        if self.tickers is None:
            self.tickers = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","XOM","GLD"]

CFG = Config()
