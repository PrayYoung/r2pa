from __future__ import annotations

import argparse
import json
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from portfolio_rl_agent_lab.config import CFG
from portfolio_rl_agent_lab.data.io import load_returns
from portfolio_rl_agent_lab.features.market import compute_market_summary
from portfolio_rl_agent_lab.rl.registry import load_model
from portfolio_rl_agent_lab.env.portfolio_env import _simplex_project
from portfolio_rl_agent_lab.llm.oracle_heuristic import heuristic_from_summary
from portfolio_rl_agent_lab.llm.oracle_local import local_regime_from_summary
from portfolio_rl_agent_lab.llm.store import load_regime_store
from portfolio_rl_agent_lab.text.news_alpaca import fetch_alpaca_news_daily
from portfolio_rl_agent_lab.text.select_bullets import select_representative_bullets
from portfolio_rl_agent_lab.text.encoder import NewsEncoder


def _parse_weights(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_tickers(s: str) -> List[str]:
    return [x.strip().upper() for x in s.split(",") if x.strip()]


def _build_observation(
    returns: pd.DataFrame,
    asof: pd.Timestamp,
    current_weights: np.ndarray,
    use_regime: bool,
    regime_dim: int,
    regime_row: Optional[np.ndarray] = None,
) -> np.ndarray:
    idx = returns.index
    if asof not in idx:
        raise ValueError(f"asof date not found in returns index: {asof.date()}")

    t = idx.get_loc(asof)
    if t < CFG.window:
        raise ValueError(f"Not enough history before {asof.date()} for window={CFG.window}")

    window_rets = returns.iloc[t - CFG.window : t].to_numpy(dtype=np.float32)
    feat = window_rets.T.reshape(-1)
    parts = [feat, current_weights.astype(np.float32)]

    if use_regime:
        if regime_row is None:
            regime_df = load_regime_store().sort_index()
            if not isinstance(regime_df.index, pd.DatetimeIndex):
                regime_df.index = pd.to_datetime(regime_df.index)
            row = regime_df.reindex(idx).ffill().fillna(0.0).loc[asof].to_numpy(dtype=np.float32)
        else:
            row = regime_row.astype(np.float32)
        if row.shape[0] != regime_dim:
            raise ValueError(f"Regime dim mismatch: got {row.shape[0]}, expected {regime_dim}")
        parts.append(row)

    obs = np.concatenate(parts, axis=0).astype(np.float32)
    return obs


def _fetch_live_returns(
    tickers: List[str],
    lookback_days: int,
) -> pd.DataFrame:
    if lookback_days <= CFG.window + 5:
        raise ValueError(f"lookback_days must be > window ({CFG.window}) + buffer.")
    df = yf.download(
        tickers,
        period=f"{lookback_days}d",
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise ValueError("Yahoo download returned empty data.")
    prices = df["Close"].copy()
    prices = prices.dropna(how="all").ffill().dropna()
    rets = prices.pct_change().dropna()
    if rets.empty:
        raise ValueError("Computed returns are empty.")
    return rets


def _fetch_live_news_alpaca(
    symbols: List[str],
    start_date: str,
    end_date: str,
    include_content: bool = False,
) -> dict:
    return fetch_alpaca_news_daily(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        include_content=include_content,
    )


def run_policy(
    model_path: str,
    algo: str,
    returns_path: str,
    asof: Optional[str],
    current_weights: Optional[np.ndarray],
    use_regime: bool,
    regime_source: Optional[str],
    out_path: Optional[str],
    live_yahoo: bool,
    tickers: Optional[List[str]],
    lookback_days: int,
    live_news: bool,
    news_lookback_days: int,
    news_include_content: bool,
) -> dict:
    CFG.regime_source = regime_source or "heuristic"

    if not use_regime:
        CFG.use_regime_features = False

    if live_yahoo:
        tickers = tickers or CFG.tickers
        returns = _fetch_live_returns(tickers, lookback_days=lookback_days)
        if live_news and getattr(CFG, "regime_source", None) != "local":
            print("[infer] live_news is enabled but regime_source is not 'local'; ignoring live_news.")
        if CFG.use_regime_features and getattr(CFG, "regime_source", None) not in {"heuristic", "local"}:
            print(
                "[infer] live_yahoo enabled but regime_source is neither 'heuristic' nor 'local'; "
                "falling back to stored regime features."
            )
    else:
        returns = load_returns(returns_path)
    returns.index = pd.to_datetime(returns.index)

    asof_ts = pd.to_datetime(asof) if asof else returns.index[-1]

    n_assets = returns.shape[1]
    n_action = n_assets + (1 if CFG.cash_asset else 0)

    if current_weights is None:
        current_weights = np.ones(n_action, dtype=np.float32) / n_action
    else:
        if len(current_weights) != n_action:
            raise ValueError(f"current_weights length {len(current_weights)} != n_action {n_action}")
        current_weights = _simplex_project(np.array(current_weights, dtype=np.float32))

    regime_row = None
    if live_yahoo and CFG.use_regime_features and getattr(CFG, "regime_source", None) == "heuristic":
        idx = returns.index
        t = idx.get_loc(asof_ts)
        if t < CFG.window:
            raise ValueError(f"Not enough history before {asof_ts.date()} for window={CFG.window}")
        window_rets = returns.iloc[t - CFG.window : t].to_numpy(dtype=np.float32)
        summary = compute_market_summary(window_rets)
        regime_row = np.array(heuristic_from_summary(summary).to_vector(), dtype=np.float32)
    elif live_yahoo and live_news and CFG.use_regime_features and getattr(CFG, "regime_source", None) == "local":
        idx = returns.index
        t = idx.get_loc(asof_ts)
        if t < CFG.window:
            raise ValueError(f"Not enough history before {asof_ts.date()} for window={CFG.window}")
        window_rets = returns.iloc[t - CFG.window : t].to_numpy(dtype=np.float32)
        summary = compute_market_summary(window_rets)

        symbols = tickers or CFG.tickers
        end_date = asof_ts.strftime("%Y-%m-%d")
        start_date = (asof_ts - pd.Timedelta(days=news_lookback_days)).strftime("%Y-%m-%d")
        daily = _fetch_live_news_alpaca(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            include_content=news_include_content,
        )
        texts = daily.get(end_date, [])
        encoder = NewsEncoder(model_name="all-MiniLM-L6-v2", device="cpu")
        emb = encoder.encode(texts)
        bullets = select_representative_bullets(texts, emb, k=min(3, len(texts))) if texts else []

        rf = local_regime_from_summary(
            summary=summary,
            news_bullets=bullets,
            backend="ollama",
            model="llama3:latest",
            temperature=0.0,
        )
        regime_row = np.array(rf.to_vector(), dtype=np.float32)

    obs = _build_observation(
        returns=returns,
        asof=asof_ts,
        current_weights=current_weights,
        use_regime=CFG.use_regime_features,
        regime_dim=CFG.regime_dim,
        regime_row=regime_row,
    )

    model = load_model(model_path, algo=algo)
    action, _ = model.predict(obs, deterministic=True)
    w = _simplex_project(np.array(action, dtype=np.float32))

    asset_names = list(returns.columns)
    if CFG.cash_asset:
        asset_names.append("CASH")

    weights = {name: float(w[i]) for i, name in enumerate(asset_names)}
    result = {
        "asof": asof_ts.strftime("%Y-%m-%d"),
        "model_path": model_path,
        "algo": algo,
        "regime_source": getattr(CFG, "regime_source", None),
        "use_regime": bool(CFG.use_regime_features),
        "tickers": list(returns.columns),
        "weights": weights,
    }

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run policy inference for a single date")
    parser.add_argument("--model", default="artifacts/models/ppo_portfolio", help="Path to trained RL model")
    parser.add_argument("--algo", choices=["ppo", "a2c", "sac", "td3"], default="ppo", help="RL algorithm type")
    parser.add_argument("--returns", default="artifacts/data/processed/returns.parquet", help="Returns parquet")
    parser.add_argument("--asof", default=None, help="YYYY-MM-DD (default: last date in returns)")
    parser.add_argument("--current-weights", default=None, help="Comma list of current weights")
    parser.add_argument("--no-regime", action="store_true", help="Disable regime features")
    parser.add_argument("--regime-source", default="heuristic", help="Regime source: heuristic|local|student")
    parser.add_argument("--out", default=None, help="Write output JSON to path")
    parser.add_argument("--live-yahoo", action="store_true", help="Fetch latest prices from Yahoo for inference")
    parser.add_argument("--tickers", default=None, help="Comma list of tickers (defaults to CFG.tickers)")
    parser.add_argument("--lookback-days", type=int, default=180, help="Yahoo lookback days (must cover window)")
    parser.add_argument("--live-news", action="store_true", help="Fetch live news for local LLM regime")
    parser.add_argument("--news-lookback-days", type=int, default=5, help="News lookback days")
    parser.add_argument("--news-include-content", action="store_true", help="Include full news content when available")

    args = parser.parse_args()
    cw = _parse_weights(args.current_weights) if args.current_weights else None

    tickers = _parse_tickers(args.tickers) if args.tickers else None
    result = run_policy(
        model_path=args.model,
        algo=args.algo,
        returns_path=args.returns,
        asof=args.asof,
        current_weights=cw,
        use_regime=not args.no_regime,
        regime_source=args.regime_source,
        out_path=args.out,
        live_yahoo=args.live_yahoo,
        tickers=tickers,
        lookback_days=args.lookback_days,
        live_news=args.live_news,
        news_lookback_days=args.news_lookback_days,
        news_include_content=args.news_include_content,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
