"""Build an inference JSONL file from a directory of plain-text documents.

Developer: Xinzhi Yao.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    metadata: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            doc_id = row.get("doc_id") or row.get("document_id") or row.get("id")
            if doc_id:
                metadata[str(doc_id)] = dict(row)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--glob", default="*.txt")
    args = parser.parse_args()

    if not args.documents_dir.is_dir():
        raise FileNotFoundError(f"Document directory not found: {args.documents_dir}")

    metadata = read_metadata(args.metadata_csv)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for doc_path in sorted(args.documents_dir.glob(args.glob)):
            if "documents-metadata" in doc_path.name:
                continue
            doc_id = doc_path.stem
            record = {
                "doc_id": doc_id,
                "text": doc_path.read_text(encoding="utf-8").strip(),
            }
            if doc_id in metadata:
                record["metadata"] = metadata[doc_id]
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(json.dumps({"documents": count, "output": str(args.output_jsonl)}, indent=2))


if __name__ == "__main__":
    main()
