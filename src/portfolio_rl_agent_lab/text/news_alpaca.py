from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable
import os
import time
import json
import requests
import pandas as pd


@dataclass
class AlpacaNewsConfig:
    base_url: str = "https://data.alpaca.markets"
    api_version: str = "v1beta1"
    page_limit: int = 50          # Alpaca supports pagination; 50 is a safe default
    sleep_s: float = 0.25         # gentle rate limiting


def _auth_headers() -> Dict[str, str]:
    """
    Alpaca Market Data APIs require APCA-API-KEY-ID and APCA-API-SECRET-KEY headers.
    """
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID_PAPER") or os.getenv("ALPACA_API_KEY")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY_PAPER") or os.getenv("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(
            "Missing Alpaca credentials. Set env vars APCA_API_KEY_ID and APCA_API_SECRET_KEY "
            "(or APCA_API_KEY_ID_PAPER / APCA_API_SECRET_KEY_PAPER)."
        )
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _iter_news_pages(
    symbols: List[str],
    start: str,
    end: str,
    include_content: bool,
    cfg: AlpacaNewsConfig,
) -> Iterable[dict]:
    """
    Yield raw Alpaca news articles by paging through /v1beta1/news.
    """
    url = f"{cfg.base_url}/{cfg.api_version}/news"
    headers = _auth_headers()

    params = {
        "symbols": ",".join(symbols),
        "start": start,
        "end": end,
        "limit": cfg.page_limit,
        "include_content": str(include_content).lower(),
    }

    page_token: Optional[str] = None
    while True:
        if page_token:
            params["page_token"] = page_token
        else:
            params.pop("page_token", None)

        r = requests.get(url, headers=headers, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()

        # Alpaca response typically includes `news` list and optional `next_page_token`
        items = data.get("news", []) or data.get("items", [])
        for it in items:
            yield it

        page_token = data.get("next_page_token") or data.get("next_page_token".upper())
        if not page_token:
            break

        time.sleep(cfg.sleep_s)


def fetch_alpaca_news_daily(
    symbols: List[str],
    start_date: str,
    end_date: str,
    include_content: bool = False,
    cfg: Optional[AlpacaNewsConfig] = None,
) -> Dict[str, List[str]]:
    """
    Fetch Alpaca news and aggregate into daily bullets.

    Returns: dict date_str -> list of bullet strings
    date_str format: YYYY-MM-DD
    """
    cfg = cfg or AlpacaNewsConfig()

    # Fetch raw articles
    daily: Dict[str, List[str]] = {}
    seen = set()

    for art in _iter_news_pages(
        symbols=symbols,
        start=start_date,
        end=end_date,
        include_content=include_content,
        cfg=cfg,
    ):
        # created_at is usually ISO8601; fall back to updated_at
        ts = art.get("created_at") or art.get("updated_at")
        if not ts:
            continue

        dt = pd.to_datetime(ts, utc=True).tz_convert(None)  # convert to naive UTC
        date_str = dt.strftime("%Y-%m-%d")

        headline = (art.get("headline") or "").strip()
        summary = (art.get("summary") or "").strip()
        content = (art.get("content") or "").strip()

        # Build a single bullet string
        bullet = headline
        if include_content and content:
            bullet = f"{headline} — {content}"
        elif summary:
            bullet = f"{headline} — {summary}"

        bullet = bullet.strip(" -—\n\t")
        if not bullet:
            continue

        # Deduplicate by (date, bullet)
        key = (date_str, bullet)
        if key in seen:
            continue
        seen.add(key)

        daily.setdefault(date_str, []).append(bullet)

    return daily


def save_news_daily_jsonl(
    daily: Dict[str, List[str]],
    out_path: str,
    symbols: List[str],
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for date_str in sorted(daily.keys()):
            obj = {"date": date_str, "symbols": symbols, "bullets": daily[date_str]}
            f.write(json.dumps(obj) + "\n")
