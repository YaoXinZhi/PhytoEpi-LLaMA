"""JSON helpers for schema-constrained extraction outputs.

Developer: Xinzhi Yao.
"""

from __future__ import annotations

import json
import re
from typing import Any

try:
    import json5
except Exception:  # pragma: no cover - optional dependency
    json5 = None


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
TRAILING_COMMA_RE = re.compile(r",(?=\s*[}\]])")


def strip_think_block(text: str) -> str:
    """Remove chain-of-thought wrappers from models that emit them."""

    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def balanced_json_candidates(text: str) -> list[str]:
    """Return balanced object/list substrings found in generated text."""

    pairs = {"{": "}", "[": "]"}
    openers = set(pairs)
    closers = {right: left for left, right in pairs.items()}
    candidates: list[str] = []

    for start, char in enumerate(text):
        if char not in openers:
            continue
        stack = [char]
        in_string = False
        quote = ""
        escaped = False
        for idx in range(start + 1, len(text)):
            cur = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif cur == "\\":
                    escaped = True
                elif cur == quote:
                    in_string = False
                continue
            if cur in ("'", '"'):
                in_string = True
                quote = cur
                continue
            if cur in openers:
                stack.append(cur)
                continue
            if cur in closers:
                if not stack or stack[-1] != closers[cur]:
                    break
                stack.pop()
                if not stack:
                    candidates.append(text[start : idx + 1])
                    break
    return candidates


def parse_json_like(text: str) -> Any | None:
    """Parse strict JSON first, then optional JSON5 after trailing-comma cleanup."""

    candidate = TRAILING_COMMA_RE.sub("", text.strip())
    loaders = [json.loads]
    if json5 is not None:
        loaders.append(json5.loads)
    for loader in loaders:
        try:
            return loader(candidate)
        except Exception:
            continue
    return None


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first JSON object in a generated string.

    The evaluator used in the experiments applied lightweight post-processing
    before parsing. This helper mirrors that intent without silently repairing
    arbitrary malformed structures.
    """

    clean_text = strip_think_block(text)
    candidates = [clean_text]
    candidates.extend(block.strip() for block in JSON_BLOCK_RE.findall(clean_text))
    candidates.extend(balanced_json_candidates(clean_text))

    start = clean_text.find("{")
    end = clean_text.rfind("}")
    if 0 <= start < end:
        candidates.append(clean_text[start : end + 1])

    for candidate in reversed(candidates):
        parsed = parse_json_like(candidate)
        if isinstance(parsed, dict):
            return normalize_prediction(parsed)
        if isinstance(parsed, list):
            normalized = squash_prediction_list(parsed)
            if normalized is not None:
                return normalize_prediction(normalized)
    return None


def squash_prediction_list(items: list[Any]) -> dict[str, Any] | None:
    """Convert list envelopes into the paper's dict envelope when possible."""

    entities: list[Any] = []
    relationships: list[Any] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        entities.extend(item.get("entities", []))
        relationships.extend(item.get("relationships", []))
    if not entities and not relationships:
        return None
    return {"entities": entities, "relationships": relationships}


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
