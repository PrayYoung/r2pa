from __future__ import annotations

from typing import Literal

from stable_baselines3 import A2C, PPO, SAC, TD3

AlgoName = Literal["ppo", "a2c", "sac", "td3"]


def normalize_algo(algo: str) -> AlgoName:
    a = algo.lower().strip()
    if a not in {"ppo", "a2c", "sac", "td3"}:
        raise ValueError(f"Unsupported algo: {algo}. Use one of ppo|a2c|sac|td3")
    return a  # type: ignore[return-value]


def model_class(algo: str):
    a = normalize_algo(algo)
    if a == "ppo":
        return PPO
    if a == "a2c":
        return A2C
    if a == "sac":
        return SAC
    return TD3


def load_model(model_path: str, algo: str = "ppo"):
    cls = model_class(algo)
    return cls.load(model_path)
