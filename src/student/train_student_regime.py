from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


FEATURES = ["mkt_mom", "mkt_vol", "mkt_mdd", "news_count", "news_var", "news_shift"]
TARGETS = ["y_regime", "y_conf", "y_macro_risk", "y_equity_bias", "y_defensive_bias"]


class StudentMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def standardize(train_x: np.ndarray, x: np.ndarray):
    mu = train_x.mean(axis=0, keepdims=True)
    sig = train_x.std(axis=0, keepdims=True) + 1e-8
    return (x - mu) / sig, mu, sig


def main():
    os.makedirs("artifacts/models", exist_ok=True)

    df = pd.read_parquet("artifacts/data/processed/student_dataset.parquet").sort_index()

    # Simple time split: last 20% as validation
    n = len(df)
    n_train = int(n * 0.8)

    x = df[FEATURES].to_numpy(dtype=np.float32)
    y = df[TARGETS].to_numpy(dtype=np.float32)

    x_train, x_val = x[:n_train], x[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    x_train_std, mu, sig = standardize(x_train, x_train)
    x_val_std = (x_val - mu) / sig

    train_ds = TensorDataset(torch.from_numpy(x_train_std), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val_std), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StudentMLP(in_dim=x.shape[1], out_dim=y.shape[1]).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, 51):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())

        tr = float(np.mean(train_losses))
        va = float(np.mean(val_losses))
        print(f"Epoch {epoch:03d} | train_mse={tr:.6f} | val_mse={va:.6f}")

        if va < best_val:
            best_val = va
            ckpt = {
                "state_dict": model.state_dict(),
                "mu": mu.astype(np.float32),
                "sig": sig.astype(np.float32),
                "features": FEATURES,
                "targets": TARGETS,
            }
            torch.save(ckpt, "artifacts/models/student_regime.pt")

    print("Saved: artifacts/models/student_regime.pt")


if __name__ == "__main__":
    main()
