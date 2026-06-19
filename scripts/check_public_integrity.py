"""Check public-release hygiene for the repository.

Developer: Xinzhi Yao.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SECRET_PATTERNS = {
    "openai_key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "huggingface_token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "generic_private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "bearer_token": re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}"),
}

LOCAL_PATH_MARKERS = (
    "/mnt/" + "beegfs/",
    "/" + "Users/",
    "Nutstore" + " Files",
    "INRAE" + "留学",
)

TEXT_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".toml", ".txt", ".jsonl", ".gitignore"}


def iter_text_files(root: Path):
    for path in root.rglob("*"):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.is_file() and (path.suffix in TEXT_SUFFIXES or path.name == ".gitignore"):
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()

    root = args.root.resolve()
    failures: list[str] = []

    for path in iter_text_files(root):
        rel = path.relative_to(root)
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{rel}: matched secret pattern `{name}`")
        for marker in LOCAL_PATH_MARKERS:
            if marker in text:
                failures.append(f"{rel}: contains local path marker `{marker}`")

    for path in sorted((root / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "Developer: Xinzhi Yao" not in text:
            failures.append(f"{path.relative_to(root)}: missing Developer attribution")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        return 1

    print("public integrity check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
