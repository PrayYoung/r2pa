from __future__ import annotations

import json
import re
from typing import Any, Dict


def extract_first_json_object(text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from a model output.
    This makes the system robust to extra tokens like markdown fences or prose.
    """
    text = text.strip()

    # Fast path: pure JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Remove common markdown fences
    text = re.sub(r"^```(json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()

    # Find first {...} block (naive but effective)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in text:\n{text}")

    candidate = text[start : end + 1]
    return json.loads(candidate)
