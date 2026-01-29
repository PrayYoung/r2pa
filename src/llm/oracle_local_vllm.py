from __future__ import annotations

import json
from typing import Dict, Optional, List

import requests

from src.llm.schema import RegimeFeatures

def _map_regime(s: str) -> int:
    return {"risk_off": -1, "neutral": 0, "risk_on": 1}[s]

def vllm_regime_from_summary(
    summary: Dict,
    news_bullets: Optional[List[str]] = None,
    base_url: str = "http://127.0.0.1:8000/v1",
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> RegimeFeatures:
    """
    Query a local vLLM OpenAI-compatible endpoint to produce regime features as JSON.
    This function is deterministic by default (temperature=0).
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
""".strip()

    req = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return JSON only. No prose."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    r = requests.post(f"{base_url}/chat/completions", json=req, timeout=120)
    r.raise_for_status()
    out = r.json()

    raw = out["choices"][0]["message"]["content"].strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Local model did not return valid JSON:\n{raw}") from e

    return RegimeFeatures(
        regime=_map_regime(data["regime"]),
        confidence=float(data["confidence"]),
        macro_risk=float(data["macro_risk"]),
        equity_bias=float(data["equity_bias"]),
        defensive_bias=float(data["defensive_bias"]),
    )
