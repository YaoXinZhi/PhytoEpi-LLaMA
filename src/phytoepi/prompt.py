"""Prompt schema used for plant-health relation extraction.

Developer: Xinzhi Yao.
"""

from __future__ import annotations

from phytoepi.schema import ENTITY_TYPES, RELATION_TYPES

__all__ = ["ENTITY_TYPES", "RELATION_TYPES", "SCHEMA_PROMPT", "build_extraction_prompt"]

SCHEMA_PROMPT = """Identify all named entities and relationships from the text, adhering strictly to the schema below.

Entity types:
- Pest: a plant pest that can infect or damage a host plant.
- Vector: an insect that can transmit a pest to a host plant.
- Plant: a plant.
- Disease: a plant disease.
- Geographic: a political or physical location.

Relationship types and argument constraints:
- Located in: source type in [Plant, Pest, Disease], target type Geographic.
- Causes: source type Pest, target type Disease.
- Have been found on: source type in [Pest, Vector], target type Plant.
- Affects: source type Disease, target type Plant.
- Transmits: source type Vector, target type Pest.

Return a single JSON object with exactly two top-level keys:
{
  "entities": [{"type": "...", "name": "..."}],
  "relationships": [{"source": "...", "type": "...", "target": "..."}]
}

Constraints:
- Entity names must appear verbatim in the text.
- Do not include generic mentions when a more specific entity is available.
- Relationship arguments must refer to extracted entity names.
- Extract hypothetical, uncertain, or negated relations when they are expressed in the text.
- If no entities or relationships are found, return empty lists.

Text:
{text}

JSON:
"""


def build_extraction_prompt(text: str) -> str:
    """Return the schema-constrained extraction prompt for one document."""

    return SCHEMA_PROMPT.format(text=text.strip())
