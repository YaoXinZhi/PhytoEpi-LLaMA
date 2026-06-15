"""Run schema-constrained plant-health relation extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phytoepi.json_utils import extract_json_object
from phytoepi.prompt import build_extraction_prompt


def read_documents(path: Path, id_field: str, text_field: str):
    with path.open() as handle:
        for i, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            yield str(record.get(id_field, i)), str(record[text_field])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--documents-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--id-field", default="doc_id")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=8128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--disable-eos", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map="auto")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    eos_token_id = None if args.disable_eos else tokenizer.eos_token_id
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w") as out:
        for doc_id, text in tqdm(list(read_documents(args.documents_jsonl, args.id_field, args.text_field))):
            prompt = build_extraction_prompt(text)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            for repeat_id in range(args.repeat):
                generation_kwargs = {
                    "do_sample": True,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "min_p": args.min_p,
                    "top_k": args.top_k,
                    "max_new_tokens": args.max_new_tokens,
                    "eos_token_id": eos_token_id,
                    "pad_token_id": tokenizer.eos_token_id,
                }
                with torch.no_grad():
                    generated = model.generate(**inputs, **generation_kwargs)
                output_ids = generated[0, inputs.input_ids.shape[1] :]
                text_out = tokenizer.decode(output_ids, skip_special_tokens=True)
                parsed = extract_json_object(text_out)
                out.write(
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "repeat": repeat_id,
                            "output": text_out,
                            "valid_json": parsed is not None,
                            "parsed": parsed,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
