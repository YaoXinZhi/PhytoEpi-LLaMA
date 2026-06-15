"""Continual pre-training with LoRA adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--embedding-learning-rate", type=float, default=5e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-r", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=3407)
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
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    class CptTrainer(Trainer):
        """Trainer with a lower learning rate for embedding and LM-head adapters."""

        def __init__(self, *trainer_args, embedding_learning_rate: float, **trainer_kwargs):
            self.embedding_learning_rate = embedding_learning_rate
            super().__init__(*trainer_args, **trainer_kwargs)

        def create_optimizer(self):
            if self.optimizer is not None:
                return self.optimizer
            base_params = []
            embedding_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if "embed_tokens" in name or "lm_head" in name:
                    embedding_params.append(param)
                else:
                    base_params.append(param)
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": base_params, "lr": self.args.learning_rate},
                    {"params": embedding_params, "lr": self.embedding_learning_rate},
                ],
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.args.weight_decay,
            )
            return self.optimizer

    set_seed(args.seed)
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
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "embed_tokens",
            "lm_head",
        ],
        use_rslora=True,
    )
    model = get_peft_model(model, lora_config)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch[args.text_field],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )

    train_ds = Dataset.from_list(read_jsonl(args.train_jsonl)).map(
        tokenize, batched=True, remove_columns=None
    )
    eval_ds = None
    if args.eval_jsonl:
        eval_ds = Dataset.from_list(read_jsonl(args.eval_jsonl)).map(
            tokenize, batched=True, remove_columns=None
        )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=500,
        eval_steps=500 if eval_ds is not None else None,
        evaluation_strategy="steps" if eval_ds is not None else "no",
        bf16=args.bf16,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = CptTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        embedding_learning_rate=args.embedding_learning_rate,
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
