"""Evaluate schema-constrained EPOP relation extraction predictions.

Developer: Xinzhi Yao.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from phytoepi.json_utils import extract_json_object
from phytoepi.schema import (
    EVAL_ENTITY_TYPES,
    RELATION_ARGUMENT_TYPES,
    RELATION_TYPES,
    canonical_entity_type,
    canonical_relation_type,
    normalize_name,
    normalize_relation_type,
)

NORM_TYPES = ("NCBI_Taxonomy", "GeoNames", "OntoBiotope", "name")


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        self.parent[self.find(right)] = self.find(left)


def read_json_like(path: Path) -> Any:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        parsed = extract_json_object(text)
        if parsed is None:
            raise
        return parsed


def squash_annotation(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if not isinstance(obj, list):
        return {"entities": [], "relationships": []}
    entities: list[Any] = []
    relationships: list[Any] = []
    equivalences: list[Any] = []
    for item in obj:
        if not isinstance(item, dict):
            continue
        entities.extend(item.get("entities", []))
        relationships.extend(item.get("relationships", []))
        equivalences.extend(item.get("equivalences", []))
    return {"entities": entities, "relationships": relationships, "equivalences": equivalences}


def entity_key(entity: dict[str, Any]) -> tuple[str, str, str]:
    entity_type = canonical_entity_type(entity.get("type", ""))
    for norm_type in NORM_TYPES:
        if norm_type == "name":
            return (entity_type, "name", normalize_name(entity.get("name", "")))
        if norm_type in entity and entity.get(norm_type):
            return (entity_type, norm_type, str(entity[norm_type]))
    return (entity_type, "name", normalize_name(entity.get("name", "")))


def relation_args(relation: dict[str, Any]) -> tuple[Any, Any]:
    if "arguments" in relation and isinstance(relation["arguments"], dict):
        return relation["arguments"].get("source"), relation["arguments"].get("target")
    return relation.get("source"), relation.get("target", relation.get("name"))


def load_reference(path: Path) -> list[dict[str, Any]]:
    data = squash_annotation(read_json_like(path))
    entities = [item for item in data.get("entities", []) if isinstance(item, dict)]
    relationships = [item for item in data.get("relationships", []) if isinstance(item, dict)]

    uf = UnionFind()
    key_to_id: dict[tuple[str, str, str], str] = {}
    for entity in entities:
        entity_id = str(entity.get("id", ""))
        if not entity_id:
            continue
        uf.add(entity_id)
        key = entity_key(entity)
        if key in key_to_id:
            uf.union(key_to_id[key], entity_id)
        else:
            key_to_id[key] = entity_id

    for group in data.get("equivalences", []):
        ids = [str(item) for item in group if item]
        if not ids:
            continue
        first = ids[0]
        for entity_id in ids[1:]:
            uf.union(first, entity_id)

    groups: dict[str, dict[str, Any]] = {}
    for entity in entities:
        entity_id = str(entity.get("id", ""))
        if not entity_id:
            continue
        root = uf.find(entity_id)
        group = groups.setdefault(
            root,
            {
                "ids": set(),
                "names": set(),
                "type": canonical_entity_type(entity.get("type", "")),
            },
        )
        group["ids"].add(entity_id)
        if entity.get("name"):
            group["names"].add(str(entity["name"]))

    id_to_group = {
        entity_id: group
        for group in groups.values()
        for entity_id in group["ids"]
    }

    relations: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for relation in relationships:
        source_id, target_id = relation_args(relation)
        source = id_to_group.get(str(source_id))
        target = id_to_group.get(str(target_id))
        if not source or not target:
            continue
        key = (
            canonical_relation_type(relation.get("type", "")),
            tuple(sorted(source["ids"])),
            tuple(sorted(target["ids"])),
        )
        if key in seen:
            continue
        seen.add(key)
        relations.append(
            {
                "type": canonical_relation_type(relation.get("type", "")),
                "source_names": sorted(source["names"]),
                "target_names": sorted(target["names"]),
                "source_type": source.get("type", ""),
                "target_type": target.get("type", ""),
            }
        )
    return relations


def prediction_from_obj(obj: dict[str, Any] | None) -> tuple[list[dict[str, Any]], Counter[str]]:
    if obj is None:
        return [], Counter()

    entities = obj.get("entities", [])
    relationships = obj.get("relationships", [])
    if not isinstance(entities, list):
        entities = []
    if not isinstance(relationships, list):
        relationships = []

    entity_types: dict[str, str] = {}
    invalid = Counter()
    valid_entity_types = set(EVAL_ENTITY_TYPES) | {"Geographic"}
    for entity in entities:
        if not isinstance(entity, dict):
            invalid["invalid_entity_object"] += 1
            continue
        name = str(entity.get("name", ""))
        entity_type = canonical_entity_type(entity.get("type", ""))
        if not name or not entity_type:
            invalid["entity_missing_required_key"] += 1
            continue
        if entity_type not in {canonical_entity_type(t) for t in valid_entity_types}:
            invalid["invalid_entity_type"] += 1
        entity_types[normalize_name(name)] = entity_type

    predictions: list[dict[str, Any]] = []
    for relation in relationships:
        if not isinstance(relation, dict):
            invalid["invalid_relationship_object"] += 1
            continue
        if not all(key in relation for key in ("source", "type", "target")):
            invalid["relationship_missing_required_key"] += 1
            continue
        rel_type = canonical_relation_type(relation["type"])
        source = str(relation["source"])
        target = str(relation["target"])
        source_type = entity_types.get(normalize_name(source), "")
        target_type = entity_types.get(normalize_name(target), "")
        if rel_type not in RELATION_TYPES:
            invalid["invalid_relation_type"] += 1
        elif source_type and target_type:
            allowed_source, allowed_target = RELATION_ARGUMENT_TYPES[rel_type]
            if source_type not in allowed_source or target_type not in allowed_target:
                invalid["invalid_relation_argument_type"] += 1
        predictions.append(
            {
                "type": rel_type,
                "source": source,
                "target": target,
                "source_type": source_type,
                "target_type": target_type,
            }
        )
    return predictions, invalid


def load_prediction_file(path: Path) -> tuple[dict[str, Any] | None, str]:
    if not path.exists():
        return None, "missing"
    text = path.read_text(encoding="utf-8", errors="replace")
    parsed = extract_json_object(text)
    if parsed is None:
        return None, "malformed_json"
    return parsed, "valid"


def load_prediction_jsonl(path: Path, repeat_policy: str) -> dict[str, tuple[dict[str, Any] | None, str]]:
    by_doc: dict[str, list[tuple[int, dict[str, Any] | None, str]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
            doc_id = str(row.get("doc_id", row.get("id", line_no)))
            repeat_id = int(row.get("repeat", len(by_doc[doc_id])))
            parsed = row.get("parsed")
            if not isinstance(parsed, dict):
                parsed = extract_json_object(str(row.get("output", "")))
            status = "valid" if isinstance(parsed, dict) else "malformed_json"
            by_doc[doc_id].append((repeat_id, parsed if isinstance(parsed, dict) else None, status))

    selected: dict[str, tuple[dict[str, Any] | None, str]] = {}
    for doc_id, rows in by_doc.items():
        rows.sort(key=lambda item: item[0])
        if repeat_policy == "first-valid":
            chosen = next((row for row in rows if row[2] == "valid"), rows[0])
        elif repeat_policy == "first":
            chosen = rows[0]
        else:
            raise ValueError(f"Unsupported repeat policy: {repeat_policy}")
        selected[doc_id] = (chosen[1], chosen[2])
    return selected


def matches(ref_rel: dict[str, Any], pred_rel: dict[str, Any]) -> bool:
    if normalize_relation_type(ref_rel["type"]) != normalize_relation_type(pred_rel["type"]):
        return False
    pred_source = normalize_name(pred_rel["source"])
    pred_target = normalize_name(pred_rel["target"])
    ref_sources = {normalize_name(name) for name in ref_rel["source_names"]}
    ref_targets = {normalize_name(name) for name in ref_rel["target_names"]}
    return pred_source in ref_sources and pred_target in ref_targets


def score_document(ref_rels: list[dict[str, Any]], pred_rels: list[dict[str, Any]]):
    used_predictions: set[int] = set()
    tp = 0
    for ref_rel in ref_rels:
        match_index = None
        for idx, pred_rel in enumerate(pred_rels):
            if idx in used_predictions:
                continue
            if matches(ref_rel, pred_rel):
                match_index = idx
                break
        if match_index is not None:
            used_predictions.add(match_index)
            tp += 1
    fp = len(pred_rels) - tp
    fn = len(ref_rels) - tp
    return tp, fp, fn


def filter_by_relation_type(relations: list[dict[str, Any]], relation_type: str):
    target = normalize_relation_type(relation_type)
    return [
        relation
        for relation in relations
        if normalize_relation_type(relation.get("type", "")) == target
    ]


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def macro(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    if not rows:
        return {f"{prefix}_precision": 0.0, f"{prefix}_recall": 0.0, f"{prefix}_f1": 0.0}
    return {
        f"{prefix}_precision": sum(float(row["precision"]) for row in rows) / len(rows),
        f"{prefix}_recall": sum(float(row["recall"]) for row in rows) / len(rows),
        f"{prefix}_f1": sum(float(row["f1"]) for row in rows) / len(rows),
    }


def f4(value: float) -> str:
    return f"{value:.4f}"


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> None:
    reference = {
        path.stem: load_reference(path)
        for path in sorted(args.reference_json_dir.glob("*.json"))
    }
    if not reference:
        raise ValueError(f"No reference JSON files found in {args.reference_json_dir}.")

    if args.prediction_jsonl:
        predictions = load_prediction_jsonl(args.prediction_jsonl, args.repeat_policy)
    else:
        predictions = {
            doc_id: load_prediction_file(args.prediction_json_dir / f"{doc_id}.json")
            for doc_id in reference
        }

    per_doc_rows: list[dict[str, Any]] = []
    parseable_rows: list[dict[str, Any]] = []
    end2end_rows: list[dict[str, Any]] = []
    invalid_counter = Counter()
    relation_counts = {
        (mode, relation_type): Counter({"tp": 0, "fp": 0, "fn": 0})
        for mode in ("parseable", "end2end")
        for relation_type in RELATION_TYPES
    }

    for doc_id, ref_rels in reference.items():
        pred_obj, status = predictions.get(doc_id, (None, "missing"))
        pred_rels, invalid = prediction_from_obj(pred_obj)
        invalid_counter.update(invalid)
        valid_json = status == "valid"

        if valid_json:
            tp, fp, fn = score_document(ref_rels, pred_rels)
            precision, recall, f1 = prf(tp, fp, fn)
            for relation_type in RELATION_TYPES:
                rtp, rfp, rfn = score_document(
                    filter_by_relation_type(ref_rels, relation_type),
                    filter_by_relation_type(pred_rels, relation_type),
                )
                relation_counts[("parseable", relation_type)].update(
                    {"tp": rtp, "fp": rfp, "fn": rfn}
                )
                relation_counts[("end2end", relation_type)].update(
                    {"tp": rtp, "fp": rfp, "fn": rfn}
                )
            parse_row = {
                "doc_id": doc_id,
                "mode": "parseable",
                "valid_json": 1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            parseable_rows.append(parse_row)
            per_doc_rows.append(parse_row)
        else:
            tp, fp, fn = 0, 0, len(ref_rels)
            precision, recall, f1 = 0.0, 0.0, 0.0
            for relation_type in RELATION_TYPES:
                relation_counts[("end2end", relation_type)].update(
                    {"fn": len(filter_by_relation_type(ref_rels, relation_type))}
                )

        end_row = {
            "doc_id": doc_id,
            "mode": "end2end",
            "valid_json": int(valid_json),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        end2end_rows.append(end_row)
        per_doc_rows.append(end_row)

    summary = {
        "documents": len(reference),
        "valid_json": sum(1 for row in end2end_rows if row["valid_json"]),
        "invalid_or_missing_json": sum(1 for row in end2end_rows if not row["valid_json"]),
        **macro(parseable_rows, "parseable_macro"),
        **macro(end2end_rows, "end2end_macro"),
    }
    summary.update({f"schema_{key}": value for key, value in sorted(invalid_counter.items())})

    summary_rows = [{key: f4(value) if isinstance(value, float) else value for key, value in summary.items()}]
    write_tsv(args.summary_tsv, summary_rows, list(summary_rows[0]))
    write_tsv(
        args.per_doc_tsv,
        [
            {
                **row,
                "precision": f4(float(row["precision"])),
                "recall": f4(float(row["recall"])),
                "f1": f4(float(row["f1"])),
            }
            for row in per_doc_rows
        ],
        ["doc_id", "mode", "valid_json", "tp", "fp", "fn", "precision", "recall", "f1"],
    )
    if args.per_relation_tsv:
        relation_rows = []
        for mode in ("parseable", "end2end"):
            for relation_type in RELATION_TYPES:
                counts = relation_counts[(mode, relation_type)]
                precision, recall, f1 = prf(counts["tp"], counts["fp"], counts["fn"])
                relation_rows.append(
                    {
                        "mode": mode,
                        "relation_type": relation_type,
                        "tp": counts["tp"],
                        "fp": counts["fp"],
                        "fn": counts["fn"],
                        "precision": f4(precision),
                        "recall": f4(recall),
                        "f1": f4(f1),
                    }
                )
        write_tsv(
            args.per_relation_tsv,
            relation_rows,
            ["mode", "relation_type", "tp", "fp", "fn", "precision", "recall", "f1"],
        )
    print(json.dumps(summary_rows[0], indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-json-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prediction-json-dir", type=Path)
    group.add_argument("--prediction-jsonl", type=Path)
    parser.add_argument("--summary-tsv", type=Path, required=True)
    parser.add_argument("--per-doc-tsv", type=Path, required=True)
    parser.add_argument("--per-relation-tsv", type=Path)
    parser.add_argument("--repeat-policy", choices=["first-valid", "first"], default="first-valid")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
