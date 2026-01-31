# Portfolio-RL-Agent-Lab

A research-oriented sandbox for building and evaluating a portfolio allocation agent trained with reinforcement learning (PPO), with an extensible “Regime Oracle” interface (heuristic / local vLLM) that produces structured regime features used by the RL policy.

## What’s in this repo

- **RL portfolio environment**: daily multi-asset allocation with transaction costs
- **Training**: PPO (Stable-Baselines3)
- **Evaluation**: backtest + benchmarks + diagnostics
- **Regime Oracle (pluggable)**:
  - heuristic oracle (rule-based)
  - local vLLM oracle (OpenAI-compatible endpoint)

## Quickstart (uv)

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

Build offline regime features (heuristic)
```bash
uv run python -m portfolio_rl_agent_lab.llm.build_regime_features
```

Train
```bash
uv run python -m portfolio_rl_agent_lab.train.train_ppo
```

Evaluate
```bash
uv run python -m portfolio_rl_agent_lab.eval.benchmarks
uv run python -m portfolio_rl_agent_lab.eval.diagnostics
```

CLI (after `uv sync`)
```bash
prl data download
prl data news-alpaca --days 5
prl rl train
prl rl benchmarks
```

## Notes

- Large artifacts are intentionally excluded from git: artifacts/, .venv/.
- The Regime Oracle is designed to be swappable without changing the RL env/policy code.
