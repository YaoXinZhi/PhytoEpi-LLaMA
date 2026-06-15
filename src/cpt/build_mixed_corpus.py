"""Build the 1B-token mixed corpus used for continual pre-training.

Input files are JSONL files with a text field. The script applies the same
lightweight filtering to plant-health and general-domain text, alternates the
cleaned records in an approximate 4:1 ratio, and stops at the requested token
budget.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from itertools import cycle
from pathlib import Path
from typing import Iterable

ABBREVIATIONS = ("e.g.", "i.e.", "Fig.", "Dr.", "Prof.", "sp.", "spp.", "cf.")
URL_RE = re.compile(r"https?://|www\.")
SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def looks_english(text: str, threshold: float = 0.75) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    ascii_letters = [c for c in letters if "a" <= c.lower() <= "z"]
    return len(ascii_letters) / max(1, len(letters)) >= threshold


def passes_quality(text: str, min_chars: int, max_digit_ratio: float, max_url_ratio: float) -> bool:
    if len(text) < min_chars:
        return False
    if not looks_english(text):
        return False
    digit_ratio = sum(c.isdigit() for c in text) / max(1, len(text))
    if digit_ratio > max_digit_ratio:
        return False
    url_hits = len(URL_RE.findall(text))
    if url_hits / max(1, len(text.split())) > max_url_ratio:
        return False
    separator_ratio = sum(c in "|=_*" for c in text) / max(1, len(text))
    return separator_ratio <= 0.05


def split_long_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    protected = text
    placeholders: dict[str, str] = {}
    for i, abbr in enumerate(ABBREVIATIONS):
        key = f"__ABBR_{i}__"
        placeholders[key] = abbr
        protected = protected.replace(abbr, key)
    parts = re.split(r"(?<=[.!?])\s+", protected)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for part in parts:
        restored = part
        for key, abbr in placeholders.items():
            restored = restored.replace(key, abbr)
        if current and current_len + len(restored) > max_chars:
            chunks.append(" ".join(current).strip())
            current = []
            current_len = 0
        current.append(restored)
        current_len += len(restored) + 1
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def iter_clean_texts(path: Path, text_field: str, args: argparse.Namespace) -> Iterable[str]:
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            text = normalize_text(str(record.get(text_field, "")))
            if not passes_quality(text, args.min_chars, args.max_digit_ratio, args.max_url_ratio):
                continue
            yield from split_long_text(text, args.max_chars)


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant-jsonl", type=Path, required=True)
    parser.add_argument("--general-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--tokenizer", default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--token-budget", type=int, default=1_000_000_000)
    parser.add_argument("--plant-per-general", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--max-digit-ratio", type=float, default=0.35)
    parser.add_argument("--max-url-ratio", type=float, default=0.02)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    plant = list(iter_clean_texts(args.plant_jsonl, args.text_field, args))
    general = list(iter_clean_texts(args.general_jsonl, args.text_field, args))
    if not plant:
        raise ValueError("No plant-health records survived filtering.")
    if not general:
        raise ValueError("No general-domain records survived filtering.")
    random.shuffle(plant)
    random.shuffle(general)

    schedule = ["plant"] * args.plant_per_general + ["general"]
    streams = {"plant": cycle(plant), "general": cycle(general)}
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    total_records = 0
    with args.output_jsonl.open("w") as out:
        while total_tokens < args.token_budget:
            for source in schedule:
                text = next(streams[source])
                n_tokens = token_len(tokenizer, text)
                if total_tokens + n_tokens > args.token_budget and total_records > 0:
                    break
                out.write(json.dumps({"text": text, "source": source}, ensure_ascii=False) + "\n")
                total_tokens += n_tokens
                total_records += 1
                if total_tokens >= args.token_budget:
                    break

    print(json.dumps({"records": total_records, "tokens": total_tokens}, indent=2))


if __name__ == "__main__":
    main()
