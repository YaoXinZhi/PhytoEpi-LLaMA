# PhytoEpi-LLaMA

Developer: Xinzhi Yao.

PhytoEpi-LLaMA is the public reference workflow for continual pre-training
(CPT), instruction fine-tuning (IFT), schema-constrained inference, dataset
conversion, and EPOP relation-extraction evaluation for plant-health literature.

This repository intentionally does not include private corpora, licensed EPOP
documents, trained weights, generated outputs, API keys, scheduler scripts, or
machine-specific paths. All data and model locations are supplied through
command-line arguments.

## Paper Alignment

| Paper component | Public implementation | Status |
| --- | --- | --- |
| CPT 1B mixed corpus, plant-health/RedPajama 4:1 | `src/cpt/build_mixed_corpus.py` | Included |
| CPT LoRA training, rank 128, sequence length 4096, packing | `src/cpt/train_cpt_lora.py` | Included |
| Held-out PPL evaluation | `src/cpt/evaluate_ppl.py` | Included |
| IFT from EPOP train + DP/BB4 train/dev | `src/data/convert_ie_pairs.py`, `src/ift/train_ift_lora.py` | Included |
| EPOP dev held out from IFT | `configs/ift_all_ie.yaml` and conversion command | Included |
| Repeated JSON extraction, 5 repeats, max 8128 tokens | `src/inference/run_schema_extraction.py` | Included |
| JSON repair/parsing and relation evaluation | `src/phytoepi/json_utils.py`, `src/eval/evaluate_epop_relations.py` | Included |
| Raw datasets/checkpoints/results | User-provided local paths | Not redistributed |

The default backbone is `unsloth/llama-3-8b-bnb-4bit`, matching the current
paper description. If you use a locally merged CPT or CPT+IFT checkpoint, pass
that path with `--model` or `--adapter`.

## Repository Layout

```text
configs/                    Paper-aligned hyperparameter snapshots
data/epop/                 EPOP public citation and dataset manifest
data/examples/              Tiny format-only examples
docs/                       Release notes and dataset documentation
scripts/check_public_integrity.py
src/cpt/                    CPT corpus mixing, training, PPL
src/data/                   IFT and inference data conversion/audit
src/eval/                   EPOP relation evaluation
src/ift/                    Response-only IFT training
src/inference/              Repeated schema-constrained extraction
src/phytoepi/               Shared schema, prompt, JSON parsing
```

## Installation

Create a Python 3.10+ environment, install PyTorch for your CUDA version, then:

```bash
pip install -r requirements.txt
```

Run scripts from the repository root with `PYTHONPATH=src`.

## Data Formats

CPT JSONL:

```json
{"text": "Plant-health or general-domain passage.", "source": "plant"}
```

IFT JSONL:

```json
{
  "instruction": "Extract entities and relationships from the provided text as JSON.",
  "input": "Document text.",
  "output": {"entities": [], "relationships": []},
  "dataset": "EPOP",
  "split": "train",
  "doc_id": "103963"
}
```

Inference JSONL:

```json
{"doc_id": "103963", "text": "Document text."}
```

## EPOP Dataset And Citation

EPOP (Epidemiomonitoring of Plants) is the plant-health benchmark used for the
paper-aligned IFT, inference, and evaluation workflow in this repository. The
LREC 2026 paper is:

> Claire Nedellec, Marine Courtin, Xinzhi Yao, Marie Grosdidier, Isabelle
> Pieretti, Sandy Duperier, and Robert Bossy. 2026. EPOP: A Benchmark Corpus
> for Assessing NLP Models on Structured Information Extraction in Plant
> Health. In LREC 2026, pp. 1331-1340. DOI:
> https://doi.org/10.63317/4in2fpefq4pz

Official EPOP resources:

| Resource | DOI / URL |
| --- | --- |
| LREC paper | https://lrec.elra.info/lrec2026-main-103 |
| EPOP documents | https://doi.org/10.57745/YKSEPY |
| EPOP train/dev annotations | https://doi.org/10.57745/ZDNOGF |
| Annotation guidelines | https://hal.science/hal-04744299 |

The EPOP documents dataset contains 247 plant-health documents split into
110 train, 55 development, and 82 test documents. The annotation dataset
contains the gold train and development labels used for information-extraction
training and validation.

Because the EPOP documents data-use terms prohibit redistribution or
unauthorized sharing, raw EPOP text is not committed to GitHub. Public citation,
URL, file, split, checksum, and local-layout metadata are included in
`CITATION.cff`, `docs/epop_dataset.md`, `data/epop/manifest.json`, and
`data/epop/citation.bib`.

## Dataset Preparation

Convert EPOP/DP/BB4 `.txt` documents and gold `.json` annotations into the IFT
format. The command below keeps EPOP development held out, while including
DP/BB4 train and development partitions in the IFT pool, as described in the
paper.

```bash
PYTHONPATH=src python src/data/convert_ie_pairs.py \
  --documents-root corpora/bionlp-st \
  --annotations-root corpora/json \
  --datasets EPOP DP BB4 \
  --splits train dev \
  --exclude EPOP:dev \
  --output-jsonl data/ift_all_ie_train.jsonl
```

Build the EPOP development inference file:

```bash
PYTHONPATH=src python src/data/build_inference_jsonl.py \
  --documents-dir data/EPOP_documents/dev \
  --output-jsonl data/epop_dev_55.jsonl
```

Audit converted IFT labels:

```bash
PYTHONPATH=src python src/data/audit_ift_jsonl.py \
  --jsonl data/ift_all_ie_train.jsonl \
  --output-tsv outputs/ift_schema_stats.tsv
```

## CPT

Build the 4:1 mixed CPT corpus:

```bash
PYTHONPATH=src python src/cpt/build_mixed_corpus.py \
  --plant-jsonl data/plant_health.jsonl \
  --general-jsonl data/redpajama.jsonl \
  --output-jsonl data/cpt_mixed_1b.jsonl \
  --tokenizer unsloth/llama-3-8b-bnb-4bit \
  --token-budget 1000000000 \
  --plant-per-general 4 \
  --seed 3407
```

Train the CPT LoRA adapter:

```bash
PYTHONPATH=src python src/cpt/train_cpt_lora.py \
  --model unsloth/llama-3-8b-bnb-4bit \
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
  --model unsloth/llama-3-8b-bnb-4bit \
  --adapter outputs/cpt_lora_1b \
  --eval-jsonl data/cpt_valid_plant_health.jsonl \
  --bf16
```

## IFT

Train the response-only IFT adapter:

```bash
PYTHONPATH=src python src/ift/train_ift_lora.py \
  --model unsloth/llama-3-8b-bnb-4bit \
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

For CPT+IFT, start IFT from a CPT-adapted checkpoint or a checkpoint produced
after merging the CPT adapter into the base model.

## Inference

Run five repeated schema-constrained generations for the 55 EPOP development
documents:

```bash
PYTHONPATH=src python src/inference/run_schema_extraction.py \
  --model unsloth/llama-3-8b-bnb-4bit \
  --adapter outputs/ift_all_ie_lora \
  --documents-jsonl data/epop_dev_55.jsonl \
  --output-jsonl outputs/epop_predictions.jsonl \
  --repeat 5 \
  --max-new-tokens 8128 \
  --temperature 0.2 \
  --top-p 0.9 \
  --top-k 50 \
  --seed 26 \
  --bf16
```

Each output row contains `doc_id`, `repeat`, raw generated text, `valid_json`,
and the parsed JSON object when parsing succeeds.

## Evaluation

Evaluate JSON reliability and document-level relation extraction. The default
repeat policy is `first-valid`, which selects the first parseable generation
for each document and falls back to the first generation when all repeats are
malformed.

```bash
PYTHONPATH=src python src/eval/evaluate_epop_relations.py \
  --reference-json-dir data/EPOP_json/dev \
  --prediction-jsonl outputs/epop_predictions.jsonl \
  --summary-tsv outputs/eval_summary.tsv \
  --per-doc-tsv outputs/eval_per_doc.tsv \
  --per-relation-tsv outputs/eval_per_relation.tsv
```

The summary reports valid JSON count, parseable-only macro P/R/F1, and
end-to-end macro P/R/F1 where malformed outputs count as extraction failures.

## Security And Release Hygiene

`.env`, raw data, `outputs/`, and `checkpoints/` are ignored by default. Only
the small example files under `data/examples/` and public metadata under
`data/epop/` are tracked. Before publishing, run:

```bash
python scripts/check_public_integrity.py
```

The check scans public text files for common API-token patterns, local absolute
paths, and missing `Developer: Xinzhi Yao` attribution in source files.
