"""Shared extraction schema and normalization helpers.

Developer: Xinzhi Yao.
"""

from __future__ import annotations

import re

ENTITY_TYPES = ("Pest", "Vector", "Plant", "Disease", "Geographic")
EVAL_ENTITY_TYPES = ("Pest", "Vector", "Plant", "Disease", "Location")

RELATION_TYPES = (
    "Located in",
    "Causes",
    "Has been found on",
    "Affects",
    "Transmits",
)

RELATION_TYPE_ALIASES = {
    "Cause": "Causes",
    "Affect": "Affects",
    "Have been found on": "Has been found on",
    "Has_been_found_on": "Has been found on",
    "Located_in": "Located in",
    "Transmit": "Transmits",
}

RELATION_ARGUMENT_TYPES = {
    "Located in": (("Plant", "Pest", "Disease"), ("Geographic", "Location")),
    "Causes": (("Pest",), ("Disease",)),
    "Has been found on": (("Pest", "Vector"), ("Plant",)),
    "Affects": (("Disease",), ("Plant",)),
    "Transmits": (("Vector",), ("Pest",)),
}

JUNK_CHARS_IN_TYPE = re.compile(r"[\s_]+")
QUOTE_CHARS = set("\"'`")


def normalize_relation_type(value: object) -> str:
    """Return a comparison key for relation labels."""

    canonical = RELATION_TYPE_ALIASES.get(str(value), str(value))
    return JUNK_CHARS_IN_TYPE.sub("", canonical).lower()


def canonical_relation_type(value: object) -> str:
    """Map a relation label to the paper's canonical EPOP label when possible."""

    normalized = normalize_relation_type(value)
    for relation_type in RELATION_TYPES:
        if normalize_relation_type(relation_type) == normalized:
            return relation_type
    return str(value)


def canonical_entity_type(value: object) -> str:
    """Normalize the Geographic/Location naming difference used in paper tables."""

    entity_type = str(value)
    if entity_type == "Location":
        return "Geographic"
    return entity_type


def normalize_name(value: object) -> str:
    """Normalize an entity surface form for exact document-level matching."""

    text = str(value or "").strip()
    if len(text) >= 2 and text[0] in QUOTE_CHARS and text[-1] in QUOTE_CHARS:
        text = text[1:-1]
    return " ".join(text.lower().split())
