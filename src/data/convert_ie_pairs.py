"""Convert IE document/annotation pairs into the shared IFT JSONL format.

Developer: Xinzhi Yao.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

DEFAULT_INSTRUCTION = "Extract entities and relationships from the provided text as JSON."


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return data


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def deduplicate_dicts(items: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        key = tuple(item.get(name) for name in keys)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def transform_annotation(annotation: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Drop non-semantic fields and map relation arguments from ids to names."""

    id_to_name = {
        str(entity["id"]): str(entity.get("name", ""))
        for entity in annotation.get("entities", [])
        if isinstance(entity, dict) and "id" in entity
    }

    entities = [
        {"type": str(entity.get("type", "")), "name": str(entity.get("name", ""))}
        for entity in annotation.get("entities", [])
        if isinstance(entity, dict) and entity.get("type") and entity.get("name")
    ]

    relationships: list[dict[str, str]] = []
    for relation in annotation.get("relationships", []):
        if not isinstance(relation, dict):
            continue
        source = relation.get("source")
        target = relation.get("target", relation.get("name"))
        relationships.append(
            {
                "source": id_to_name.get(str(source), str(source or "")),
                "type": str(relation.get("type", "")),
                "target": id_to_name.get(str(target), str(target or "")),
            }
        )

    return {
        "entities": deduplicate_dicts(entities, ("type", "name")),
        "relationships": deduplicate_dicts(relationships, ("source", "type", "target")),
    }


def parse_exclude(values: list[str]) -> set[tuple[str, str]]:
    excluded: set[tuple[str, str]] = set()
    for value in values:
        if ":" not in value:
            raise ValueError(f"Invalid --exclude value `{value}`. Use DATASET:SPLIT.")
        dataset, split = value.split(":", 1)
        excluded.add((dataset, split))
    return excluded


def iter_pairs(
    documents_root: Path,
    annotations_root: Path,
    datasets: list[str],
    splits: list[str],
    excluded: set[tuple[str, str]],
):
    for dataset in datasets:
        for split in splits:
            if (dataset, split) in excluded:
                continue
            doc_dir = documents_root / dataset / split
            ann_dir = annotations_root / dataset / split
            if not doc_dir.is_dir():
                raise FileNotFoundError(f"Document directory not found: {doc_dir}")
            if not ann_dir.is_dir():
                raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")
            for doc_path in sorted(doc_dir.glob("*.txt")):
                ann_path = ann_dir / f"{doc_path.stem}.json"
                if not ann_path.exists():
                    raise FileNotFoundError(f"Missing annotation for {doc_path}: {ann_path}")
                yield dataset, split, doc_path, ann_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents-root", type=Path, required=True)
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["EPOP", "DP", "BB4"])
    parser.add_argument("--splits", nargs="+", default=["train", "dev"])
    parser.add_argument("--exclude", action="append", default=["EPOP:dev"])
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    args = parser.parse_args()

    excluded = parse_exclude(args.exclude)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    total = 0
    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for dataset, split, doc_path, ann_path in iter_pairs(
            args.documents_root,
            args.annotations_root,
            args.datasets,
            args.splits,
            excluded,
        ):
            record = {
                "instruction": args.instruction,
                "input": read_text(doc_path),
                "output": transform_annotation(read_json(ann_path)),
                "dataset": dataset,
                "split": split,
                "doc_id": doc_path.stem,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[f"{dataset}:{split}"] = counts.get(f"{dataset}:{split}", 0) + 1
            total += 1

    print(json.dumps({"records": total, "by_source": counts}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
