"""Audit converted IFT JSONL files for dataset and schema statistics.

Developer: Xinzhi Yao.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc


def output_object(record: dict[str, Any]) -> dict[str, Any]:
    output = record.get("output", record.get("response", {}))
    if isinstance(output, str):
        return json.loads(output)
    if isinstance(output, dict):
        return output
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    args = parser.parse_args()

    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "records": 0,
            "entity_mentions": 0,
            "relationship_mentions": 0,
            "entity_types": Counter(),
            "relation_types": Counter(),
        }
    )

    for record in iter_jsonl(args.jsonl):
        dataset = str(record.get("dataset", "unknown"))
        entry = stats[dataset]
        entry["records"] += 1
        output = output_object(record)
        entities = output.get("entities", [])
        relationships = output.get("relationships", [])
        entry["entity_mentions"] += len(entities)
        entry["relationship_mentions"] += len(relationships)
        entry["entity_types"].update(str(entity.get("type", "")) for entity in entities)
        entry["relation_types"].update(str(rel.get("type", "")) for rel in relationships)

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "records",
        "entity_mentions",
        "relationship_mentions",
        "entity_types",
        "relation_types",
    ]
    with args.output_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for dataset in sorted(stats):
            entry = stats[dataset]
            writer.writerow(
                {
                    "dataset": dataset,
                    "records": entry["records"],
                    "entity_mentions": entry["entity_mentions"],
                    "relationship_mentions": entry["relationship_mentions"],
                    "entity_types": "; ".join(sorted(entry["entity_types"])),
                    "relation_types": "; ".join(sorted(entry["relation_types"])),
                }
            )

    print(json.dumps({"datasets": len(stats), "output": str(args.output_tsv)}, indent=2))


if __name__ == "__main__":
    main()
