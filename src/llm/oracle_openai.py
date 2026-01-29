from __future__ import annotations

import json
from typing import Dict, Optional, List

from openai import OpenAI
from src.llm.schema import RegimeFeatures

client = OpenAI()

def _map_regime(s: str) -> int:
    return {"risk_off": -1, "neutral": 0, "risk_on": 1}[s]

def openai_regime_from_summary(
    summary: Dict,
    news_bullets: Optional[List[str]] = None,
) -> RegimeFeatures:
    """
    Use an LLM to convert market summary (+ optional news) into structured regime features.
    The model is instructed to output JSON only.
    """

    payload = {
        "price_summary": summary,
        "news_bullets": news_bullets or [],
    }

    prompt = f"""
You are a professional portfolio risk analyst.

Given the following market information, output ONLY a valid JSON object
with the following fields:

- regime: one of ["risk_off", "neutral", "risk_on"]
- confidence: float in [0, 1]
- macro_risk: float in [0, 1]
- equity_bias: float in [0, 1]
- defensive_bias: float in [0, 1]

Rules:
- Do not include any explanation or extra text.
- The output MUST be valid JSON.
- Be conservative: if signals conflict, lower confidence.

Input:
{json.dumps(payload, indent=2)}
"""

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )

    # The SDK returns a structured response object; output_text is a convenience accessor
    raw = resp.output_text.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM did not return valid JSON:\n{raw}") from e

    return RegimeFeatures(
        regime=_map_regime(data["regime"]),
        confidence=float(data["confidence"]),
        macro_risk=float(data["macro_risk"]),
        equity_bias=float(data["equity_bias"]),
        defensive_bias=float(data["defensive_bias"]),
    )
