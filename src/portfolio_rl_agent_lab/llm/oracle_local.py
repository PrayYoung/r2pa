from __future__ import annotations

import json
from typing import Dict, Optional, List

import requests

from portfolio_rl_agent_lab.llm.schema import RegimeFeatures
from portfolio_rl_agent_lab.llm.json_utils import extract_first_json_object


def _map_regime(s: str) -> int:
    return {"risk_off": -1, "neutral": 0, "risk_on": 1}[s]


def _build_prompt(summary: Dict, news_bullets: Optional[List[str]] = None) -> str:
    payload = {
        "price_summary": summary,
        "news_bullets": news_bullets or [],
    }

    prompt = f"""
You are a professional portfolio risk analyst.

Given the following market information, output ONLY a valid JSON object with fields:
- regime: one of ["risk_off", "neutral", "risk_on"]
- confidence: float in [0, 1]
- macro_risk: float in [0, 1]
- equity_bias: float in [0, 1]
- defensive_bias: float in [0, 1]

Rules:
- Output MUST be valid JSON.
- No prose. No markdown. No code fences.
- Be conservative: if signals conflict, lower confidence.

Input:
{json.dumps(payload, indent=2)}
""".strip()

    return prompt


def _call_vllm_chat(
    prompt: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    """
    Call a vLLM OpenAI-compatible endpoint: POST {base_url}/chat/completions
    """
    req = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return JSON only. Do not include any extra text."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(f"{base_url}/chat/completions", json=req, timeout=timeout)
    r.raise_for_status()
    out = r.json()
    return out["choices"][0]["message"]["content"]


def _call_ollama_chat(
    prompt: str,
    base_url: str,
    model: str,
    temperature: float,
    timeout: int,
) -> str:
    """
    Call Ollama chat endpoint: POST {base_url}/api/chat
    Ollama runs locally by default at http://127.0.0.1:11434
    """
    req = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return JSON only. Do not include any extra text."},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": temperature},
        "stream": False,
    }
    r = requests.post(f"{base_url}/api/chat", json=req, timeout=timeout)
    r.raise_for_status()
    out = r.json()
    return out["message"]["content"]


def local_regime_from_summary(
    summary: Dict,
    news_bullets: Optional[List[str]] = None,
    backend: str = "ollama",
    model: str = "qwen2.5:7b-instruct",
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout: int = 120,
    max_retries: int = 2,
) -> RegimeFeatures:
    """
    Produce regime features using a local LLM backend (ollama or vllm).

    backend:
      - "ollama": uses /api/chat (default port 11434)
      - "vllm": uses OpenAI-compatible /v1/chat/completions (default port 8000)
    """
    backend = backend.lower().strip()
    prompt = _build_prompt(summary, news_bullets=news_bullets)

    if backend == "ollama":
        base_url = base_url or "http://127.0.0.1:11434"
    elif backend == "vllm":
        base_url = base_url or "http://127.0.0.1:8000/v1"
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'ollama' or 'vllm'.")

    last_err = None
    for _ in range(max_retries + 1):
        try:
            if backend == "ollama":
                raw = _call_ollama_chat(
                    prompt=prompt,
                    base_url=base_url,
                    model=model,
                    temperature=temperature,
                    timeout=timeout,
                )
            else:
                raw = _call_vllm_chat(
                    prompt=prompt,
                    base_url=base_url,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )

            data = extract_first_json_object(raw)

            return RegimeFeatures(
                regime=_map_regime(str(data["regime"])),
                confidence=float(data["confidence"]),
                macro_risk=float(data["macro_risk"]),
                equity_bias=float(data["equity_bias"]),
                defensive_bias=float(data["defensive_bias"]),
            )

        except Exception as e:
            last_err = e
            # Tighten constraints on retry
            prompt = (
                "Return ONLY a JSON object. No extra text.\n\n"
                + prompt
            )

    raise RuntimeError(f"Local oracle failed after retries: {last_err}")
