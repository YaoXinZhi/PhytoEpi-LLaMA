# PhytoEpi-LLaMA

PhytoEpi-LLaMA is a lightweight adaptation pipeline for schema-constrained
relation extraction from plant-health literature. The code in this repository
contains only the public CPT, IFT, and model-inference components needed to
reproduce the paper workflow. It does not include private data, checkpoints,
machine-specific paths, job scheduler scripts, or evaluation outputs.

## Method Overview

The pipeline separates two adaptation goals:

1. Continual pre-training (CPT) adapts the base Llama 3.1 8B model to a
   1B-token mixture of plant-health literature and general-domain text.
2. Instruction fine-tuning (IFT) aligns the model to the fixed extraction
   schema and teaches it to emit valid JSON.
3. Schema-constrained inference generates repeated JSON predictions for EPOP
   development documents.

The paper reports that CPT lowers plant-health held-out perplexity, but CPT
alone weakens JSON reliability. IFT restores schema-conforming output, and the
complete CPT+IFT pipeline gives the strongest completed extraction result.

## Repository Layout

```text
configs/
  cpt_1b.yaml                 Paper CPT settings
  ift_all_ie.yaml             Paper IFT settings
  inference_epop.yaml         Repeated extraction settings
src/
  cpt/
    build_mixed_corpus.py     Clean and mix CPT data
    train_cpt_lora.py         LoRA CPT training
    evaluate_ppl.py           Held-out perplexity evaluation
  ift/
    train_ift_lora.py         Response-only IFT training
  inference/
    run_schema_extraction.py  Repeated JSON extraction
  phytoepi/
    prompt.py                 Fixed extraction schema
    json_utils.py             JSON parsing helpers
```

## Installation

Create an environment with Python 3.10 or newer, install PyTorch for your CUDA
version, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The scripts are path-agnostic. Run them with `PYTHONPATH=src` from the
repository root.

## Data Formats

CPT input files are JSONL files with one text field:

```json
{"text": "Plant-health or general-domain passage."}
```

IFT input files are JSONL files with an instruction, an input text, and a gold
JSON response:

```json
{
  "instruction": "Extract plant-health entities and relations as JSON.",
  "input": "Document text.",
  "output": {"entities": [], "relationships": []}
}
```

Inference input files are JSONL files with document identifiers and text:

```json
{"doc_id": "103963", "text": "Document text."}
```

## CPT

Build a mixed CPT corpus with the 4:1 plant-health/general-domain schedule used
in the paper:

```bash
PYTHONPATH=src python src/cpt/build_mixed_corpus.py \
  --plant-jsonl data/plant_health.jsonl \
  --general-jsonl data/general_domain.jsonl \
  --output-jsonl data/cpt_mixed_1b.jsonl \
  --tokenizer meta-llama/Meta-Llama-3.1-8B \
  --token-budget 1000000000 \
  --plant-per-general 4 \
  --seed 3407
```

Train the CPT LoRA adapter:

```bash
PYTHONPATH=src python src/cpt/train_cpt_lora.py \
  --model meta-llama/Meta-Llama-3.1-8B \
  --train-jsonl data/cpt_mixed_1b.jsonl \
  --eval-jsonl data/cpt_valid_mixed.jsonl \
  --output-dir outputs/cpt_lora_1b \
  --max-seq-length 4096 \
  --learning-rate 5e-5 \
  --embedding-learning-rate 5e-6 \
  --gradient-accumulation-steps 16 \
  --lora-r 128 \
  --lora-alpha 32 \
  --seed 3407 \
  --load-in-4bit \
  --bf16
```

Evaluate held-out perplexity:

```bash
PYTHONPATH=src python src/cpt/evaluate_ppl.py \
  --model meta-llama/Meta-Llama-3.1-8B \
  --adapter outputs/cpt_lora_1b \
  --eval-jsonl data/cpt_valid_plant_health.jsonl \
  --bf16
```

## IFT

Train the response-only IFT adapter:

```bash
PYTHONPATH=src python src/ift/train_ift_lora.py \
  --model meta-llama/Meta-Llama-3.1-8B \
  --train-jsonl data/ift_all_ie_train.jsonl \
  --output-dir outputs/ift_all_ie_lora \
  --max-seq-length 8192 \
  --learning-rate 2e-5 \
  --min-learning-rate 3e-6 \
  --gradient-accumulation-steps 16 \
  --lora-r 16 \
  --seed 26 \
  --split-seed 26 \
  --dev-ratio 0.10 \
  --load-in-4bit \
  --bf16
```

For CPT+IFT, start IFT from the CPT-adapted checkpoint or merge the CPT adapter
into the base checkpoint before running the IFT command.

## Inference

Run repeated schema-constrained extraction:

```bash
PYTHONPATH=src python src/inference/run_schema_extraction.py \
  --model meta-llama/Meta-Llama-3.1-8B \
  --adapter outputs/ift_all_ie_lora \
  --documents-jsonl data/epop_dev_55.jsonl \
  --output-jsonl outputs/epop_predictions.jsonl \
  --repeat 5 \
  --max-new-tokens 8128 \
  --temperature 0.2 \
  --top-p 0.9 \
  --min-p 0.0 \
  --top-k 50 \
  --bf16
```

The output JSONL contains the raw generated text, a `valid_json` flag, and the
parsed JSON object when parsing succeeds.

## Notes

- All model paths and data paths are command-line arguments.
- The fixed extraction schema is implemented in `src/phytoepi/prompt.py`.
- The repository intentionally excludes private corpora and trained weights.
- The code is intended as a reproducible reference for the paper's CPT, IFT,
  and inference workflow.
