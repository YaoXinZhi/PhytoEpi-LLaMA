"""Held-out perplexity evaluation for causal language models."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

def iter_texts(path: Path, text_field: str):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield str(json.loads(line).get(text_field, ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--stride", type=int, default=4096)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map="auto")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    losses: list[float] = []
    tokens = 0
    with torch.no_grad():
        for text in tqdm(iter_texts(args.eval_jsonl, args.text_field), desc="ppl"):
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
            for start in range(0, len(ids), args.stride):
                chunk = ids[start : start + args.max_length]
                if len(chunk) < 2:
                    continue
                input_ids = chunk.unsqueeze(0).to(model.device)
                out = model(input_ids=input_ids, labels=input_ids)
                n = input_ids.numel() - 1
                losses.append(float(out.loss) * n)
                tokens += n

    if tokens == 0:
        raise ValueError("No tokens evaluated.")
    mean_nll = sum(losses) / tokens
    print(json.dumps({"tokens": tokens, "mean_nll": mean_nll, "ppl": math.exp(mean_nll)}, indent=2))


if __name__ == "__main__":
    main()
