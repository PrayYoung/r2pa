#!/usr/bin/env bash
set -euo pipefail

# One-command onboarding flow:
# data -> text features -> regime features -> train -> eval

ALGO="${ALGO:-ppo}"                  # ppo | a2c | sac | td3
TIMESTEPS="${TIMESTEPS:-20000}"      # keep small for fast smoke run
REGIME_SOURCE="${REGIME_SOURCE:-heuristic}"  # heuristic | local

if [[ "$REGIME_SOURCE" != "heuristic" && "$REGIME_SOURCE" != "local" ]]; then
  echo "Unsupported REGIME_SOURCE=$REGIME_SOURCE (use heuristic|local)"
  exit 1
fi

MODEL_PATH="artifacts/models/${ALGO}_portfolio"

echo "[quickstart] syncing env"
uv sync

echo "[quickstart] download market data"
uv run python -m portfolio_rl_agent_lab.data.download

echo "[quickstart] build returns"
uv run python -m portfolio_rl_agent_lab.data.make_dataset

echo "[quickstart] build text features"
uv run python -m portfolio_rl_agent_lab.text.build_text_features

if [[ "$REGIME_SOURCE" == "heuristic" ]]; then
  echo "[quickstart] build heuristic regime features"
  uv run python -m portfolio_rl_agent_lab.llm.build_regime_features
else
  echo "[quickstart] build local-LLM regime features (quick mode)"
  uv run r2pa teacher build-local --max-steps 5
fi

echo "[quickstart] train model (algo=$ALGO, timesteps=$TIMESTEPS)"
uv run r2pa rl train --algo "$ALGO" --timesteps "$TIMESTEPS"

echo "[quickstart] evaluate"
uv run r2pa rl benchmarks --algo "$ALGO" --model "$MODEL_PATH"
uv run r2pa rl diagnostics --algo "$ALGO" --model "$MODEL_PATH"

echo "[quickstart] done"
echo "Model: ${MODEL_PATH}.zip"
