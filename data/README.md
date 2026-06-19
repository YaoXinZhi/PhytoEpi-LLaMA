# Data

Developer: Xinzhi Yao.

This repository does not redistribute private corpora, EPOP documents, model
checkpoints, or generated experiment outputs. The code expects users to place
licensed datasets locally and pass their paths through command-line arguments.

Use `src/data/convert_ie_pairs.py` to convert EPOP, DP, and BB4 document/gold
annotation pairs into IFT JSONL. Use `src/data/build_inference_jsonl.py` to
convert EPOP development documents into inference JSONL. Minimal examples are
kept under `data/examples/` for smoke tests and format inspection only.
