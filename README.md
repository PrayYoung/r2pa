<p align="center">
  <img src="assets/logo.png" width="1000" />
</p>

<h1 align="center">R²PA</h1>

<p align="center">
  Regime-aware Reinforcement Learning for Portfolio Allocation
</p>

<p align="center">
  <sub>Implemented in <code>Portfolio-RL-Agent-Lab</code></sub>
</p>

A research-oriented sandbox for building and evaluating a **portfolio allocation agent** trained with **PPO**. The agent can consume **regime features** from a pluggable “Regime Oracle” (heuristic or local LLM) and is evaluated with backtests and diagnostics.

## Why this repo

- End-to-end RL workflow for daily portfolio allocation
- Pluggable regime features (heuristic or local LLM)
- Clear data → features → regime → train → eval pipeline

## Architecture (high level)

```mermaid
flowchart LR
  A[Market Data] --> B[Returns]
  B --> C[Text Features]
  C --> D[Regime Oracle]
  B --> D
  D --> E[RL Environment]
  E --> F[PPO Training]
  F --> G[Backtest & Diagnostics]
```

## Repository layout

- `portfolio_rl_agent_lab/data/` — data download + dataset building
- `portfolio_rl_agent_lab/text/` — news loading + text feature extraction
- `portfolio_rl_agent_lab/llm/` — regime feature builders (heuristic/local LLM)
- `portfolio_rl_agent_lab/student/` — student model pipeline
- `portfolio_rl_agent_lab/env/` — portfolio environment
- `portfolio_rl_agent_lab/train/` — PPO training
- `portfolio_rl_agent_lab/eval/` — backtest, benchmarks, diagnostics
- `portfolio_rl_agent_lab/pipeline/` — orchestrated pipelines
- `portfolio_rl_agent_lab/cli/` — CLI entrypoints
- `artifacts/` — generated data/models/logs (gitignored)

## Quickstart (uv)

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

### Minimal run (heuristic regime)

```bash
uv run python -m portfolio_rl_agent_lab.data.download
uv run python -m portfolio_rl_agent_lab.data.make_dataset
uv run python -m portfolio_rl_agent_lab.text.build_text_features
uv run python -m portfolio_rl_agent_lab.llm.build_regime_features
uv run python -m portfolio_rl_agent_lab.train.train_ppo
uv run python -m portfolio_rl_agent_lab.eval.benchmarks
uv run python -m portfolio_rl_agent_lab.eval.diagnostics
```

## CLI

After `uv sync`, the `prl` command is available. If you don’t want to install the script, use the module form: `uv run python -m portfolio_rl_agent_lab.cli ...`.

```bash
prl data download
prl data news-alpaca --days 5
prl rl train
prl rl benchmarks
```

## Pipeline

```bash
prl pipeline data
prl pipeline text
prl pipeline regime --source heuristic
prl pipeline student
prl pipeline rl
prl pipeline all --source heuristic
```

## Inference (single date)

```bash
prl infer run --model artifacts/models/ppo_portfolio --asof 2025-12-31
```

Live Yahoo data (latest available date)
```bash
prl infer run --live-yahoo --lookback-days 180
```

Real-time regime (heuristic)
```bash
prl infer run --live-yahoo --lookback-days 180 --regime-source heuristic
```

Real-time regime (local LLM + live news)
```bash
prl infer run --live-yahoo --live-news --regime-source local --news-lookback-days 5
```

Note: use the same ticker order as the model was trained on (defaults to `CFG.tickers`).

## Notes

- Large artifacts are excluded from git: `artifacts/`, `.venv/`
- Regime Oracle is swappable without touching env/policy logic
