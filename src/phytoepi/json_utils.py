"""JSON helpers for schema-constrained extraction outputs."""

from __future__ import annotations

import json
import re
from typing import Any


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first JSON object in a generated string.

    The evaluator used in the experiments applied lightweight post-processing
    before parsing. This helper mirrors that intent without silently repairing
    arbitrary malformed structures.
    """

    candidates = []
    block = JSON_BLOCK_RE.search(text)
    if block:
        candidates.append(block.group(1).strip())
    candidates.append(text.strip())

    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return normalize_prediction(parsed)
    return None


def normalize_prediction(obj: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Normalize a parsed prediction to the expected envelope."""

    entities = obj.get("entities", [])
    relationships = obj.get("relationships", [])
    if not isinstance(entities, list):
        entities = []
    if not isinstance(relationships, list):
        relationships = []
    return {
        "entities": [x for x in entities if isinstance(x, dict)],
        "relationships": [x for x in relationships if isinstance(x, dict)],
    }

