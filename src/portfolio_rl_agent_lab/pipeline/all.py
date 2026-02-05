from __future__ import annotations

from typing import Literal

from portfolio_rl_agent_lab.pipeline.data import run_data_pipeline
from portfolio_rl_agent_lab.pipeline.text import run_text_pipeline
from portfolio_rl_agent_lab.pipeline.regime import run_regime_pipeline
from portfolio_rl_agent_lab.pipeline.student import run_student_pipeline
from portfolio_rl_agent_lab.pipeline.rl import run_rl_pipeline

RegimeSource = Literal["heuristic", "local", "student"]


def run_all_pipeline(source: RegimeSource, algo: str = "ppo") -> None:
    run_data_pipeline()
    run_text_pipeline()

    if source == "student":
        # Teacher must exist before building student dataset
        run_regime_pipeline("heuristic")
        run_student_pipeline()
    else:
        run_regime_pipeline(source)

    run_rl_pipeline(algo=algo)


if __name__ == "__main__":
    run_all_pipeline("heuristic")
