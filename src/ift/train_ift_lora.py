"""Instruction fine-tuning with response-only loss masking.

Developer: Xinzhi Yao.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Any


RESPONSE_MARKER = "### Response:\n"
DEFAULT_MODEL = "unsloth/llama-3-8b-bnb-4bit"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
    if not records:
        raise ValueError(f"No training records found in {path}.")
    return records


def split_records(records: list[dict[str, Any]], dev_ratio: float, seed: int):
    if not records:
        raise ValueError("Cannot split an empty IFT dataset.")
    if dev_ratio <= 0 or len(records) == 1:
        return records, []
    rng = random.Random(seed)
    records = records[:]
    rng.shuffle(records)
    n_dev = min(len(records) - 1, max(1, int(round(len(records) * dev_ratio))))
    return records[n_dev:], records[:n_dev]


def build_prompt(record: dict[str, Any]) -> tuple[str, str]:
    instruction = str(record.get("instruction", "")).strip()
    text = str(record.get("input", record.get("text", ""))).strip()
    response = record.get("output", record.get("response", {}))
    if not instruction:
        raise ValueError("IFT record is missing `instruction`.")
    if not text:
        raise ValueError("IFT record is missing `input` or `text`.")
    if not isinstance(response, str):
        response = json.dumps(response, ensure_ascii=False)
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{text}\n\n{RESPONSE_MARKER}"
    return prompt, response.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--dev-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--min-learning-rate", type=float, default=3e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--dev-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--split-seed", type=int, default=26)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    records = read_jsonl(args.train_jsonl)
    if args.dev_jsonl:
        train_records = records
        dev_records = read_jsonl(args.dev_jsonl)
    else:
        train_records, dev_records = split_records(records, args.dev_ratio, args.split_seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        quantization_config=quantization_config,
        device_map="auto",
    )
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    def make_training_arguments(**kwargs):
        signature = inspect.signature(TrainingArguments.__init__)
        evaluation_strategy = kwargs.pop("evaluation_strategy")
        strategy_key = (
            "eval_strategy"
            if "eval_strategy" in signature.parameters
            else "evaluation_strategy"
        )
        kwargs[strategy_key] = evaluation_strategy
        if "lr_scheduler_kwargs" not in signature.parameters:
            kwargs.pop("lr_scheduler_kwargs", None)
        if kwargs.get("lr_scheduler_kwargs") is None:
            kwargs.pop("lr_scheduler_kwargs", None)
        return TrainingArguments(**kwargs)

    def scheduler_type_and_kwargs() -> tuple[str, dict[str, float] | None]:
        try:
            from transformers.trainer_utils import SchedulerType
        except Exception:
            return "cosine_with_min_lr", {"min_lr": args.min_learning_rate}
        supported = {item.value for item in SchedulerType}
        if "cosine_with_min_lr" in supported:
            return "cosine_with_min_lr", {"min_lr": args.min_learning_rate}
        return "cosine", None

    def tokenize(record: dict[str, Any]) -> dict[str, list[int]]:
        prompt, response = build_prompt(record)
        full_text = prompt + response + tokenizer.eos_token
        encoded = tokenizer(full_text, truncation=True, max_length=args.max_seq_length, padding=False)
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        labels = encoded["input_ids"].copy()
        labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
        encoded["labels"] = labels
        return encoded

    train_ds = Dataset.from_list(train_records).map(tokenize, remove_columns=list(train_records[0].keys()))
    dev_ds = None
    if dev_records:
        dev_ds = Dataset.from_list(dev_records).map(tokenize, remove_columns=list(dev_records[0].keys()))

    lr_scheduler_type, lr_scheduler_kwargs = scheduler_type_and_kwargs()
    training_args = make_training_arguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="steps" if dev_ds is not None else "no",
        eval_steps=100 if dev_ds is not None else None,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=dev_ds is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        bf16=args.bf16,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        ] if dev_ds is not None else None,
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
