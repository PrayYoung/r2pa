from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch

from src.config import CFG


# ---------- feature contract ----------
FEATURES = [
    "mkt_mom",
    "mkt_vol",
    "mkt_mdd",
    "news_count",
    "news_var",
    "news_shift",
]

TARGET_COLUMNS = [
    "regime",
    "confidence",
    "macro_risk",
    "equity_bias",
    "defensive_bias",
]


def compute_market_summary(window_rets: np.ndarray) -> dict:
    """
    Same market summary used during student training.
    """
    mkt = window_rets.mean(axis=1)
    mkt_mom = float(np.prod(1.0 + mkt) - 1.0)
    mkt_vol = float(np.std(mkt) * np.sqrt(252.0))
    nav = np.cumprod(1.0 + mkt)
    peak = np.maximum.accumulate(nav)
    mkt_mdd = float((nav / peak - 1.0).min())
    return {
        "mkt_mom": mkt_mom,
        "mkt_vol": mkt_vol,
        "mkt_mdd": mkt_mdd,
    }


class StudentMLP(torch.nn.Module):
    """
    Must match the architecture used in training.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def main():
    os.makedirs("artifacts/data/processed", exist_ok=True)

    # ---------- load data ----------
    rets = pd.read_parquet("artifacts/data/processed/returns.parquet")
    rets_np = rets.to_numpy(dtype=np.float32)
    idx = pd.to_datetime(rets.index)

    news = pd.read_parquet("artifacts/data/processed/news_features.parquet")
    news.index = pd.to_datetime(news.index)

    ckpt = torch.load("artifacts/models/student_regime.pt", map_location="cpu", weights_only=False)

    mu = ckpt["mu"]
    sig = ckpt["sig"]

    model = StudentMLP(in_dim=len(FEATURES), out_dim=len(TARGET_COLUMNS))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    rows = []
    dates = []

    for t in range(CFG.window, len(rets) - 2):
        dt = idx[t]

        # market features
        window_rets = rets_np[t - CFG.window:t, :]
        ms = compute_market_summary(window_rets)

        # news numeric features
        if dt in news.index:
            nf = news.loc[dt]
            news_count = float(nf.get("news_count", 0.0))
            news_var = float(nf.get("news_var", 0.0))
            news_shift = float(nf.get("news_shift", 0.0))
        else:
            news_count, news_var, news_shift = 0.0, 0.0, 0.0

        x = np.array(
            [
                ms["mkt_mom"],
                ms["mkt_vol"],
                ms["mkt_mdd"],
                news_count,
                news_var,
                news_shift,
            ],
            dtype=np.float32,
        )

        # standardize (same as training)
        mu_ = np.squeeze(mu)
        sig_ = np.squeeze(sig)
        x_std = ((x - mu_) / sig_).astype(np.float32)
        x_std = np.ravel(x_std)
        with torch.no_grad():
            y = model(torch.from_numpy(x_std).unsqueeze(0)).squeeze(0).numpy()


        # post-processing / safety clamps
        regime = float(np.clip(y[0], -1.0, 1.0))
        confidence = float(np.clip(y[1], 0.0, 1.0))
        macro_risk = float(np.clip(y[2], 0.0, 1.0))
        equity_bias = float(np.clip(y[3], 0.0, 1.0))
        defensive_bias = float(np.clip(y[4], 0.0, 1.0))

        rows.append(
            [
                regime,
                confidence,
                macro_risk,
                equity_bias,
                defensive_bias,
            ]
        )
        dates.append(dt)

    df = pd.DataFrame(
        rows,
        index=pd.to_datetime(dates),
        columns=TARGET_COLUMNS,
    )

    out_path = "artifacts/data/processed/regime_features_student.parquet"
    df.to_parquet(out_path)
    print(
        f"Saved student regime features: {out_path} | "
        f"shape={df.shape} | {df.index.min()}..{df.index.max()}"
    )


if __name__ == "__main__":
    main()
