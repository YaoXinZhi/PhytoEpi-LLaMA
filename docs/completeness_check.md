# Completeness Check

Developer: Xinzhi Yao.

This public repository was checked against the current PhytoEpi-LLaMA paper
workflow on 2026-06-19.

## Included

| Area | Files |
| --- | --- |
| CPT corpus processing | `src/cpt/build_mixed_corpus.py` |
| CPT LoRA training | `src/cpt/train_cpt_lora.py` |
| CPT held-out PPL | `src/cpt/evaluate_ppl.py` |
| IFT dataset conversion | `src/data/convert_ie_pairs.py` |
| IFT dataset audit | `src/data/audit_ift_jsonl.py` |
| IFT LoRA training | `src/ift/train_ift_lora.py` |
| EPOP inference JSONL building | `src/data/build_inference_jsonl.py` |
| Repeated schema-constrained inference | `src/inference/run_schema_extraction.py` |
| JSON parsing and schema normalization | `src/phytoepi/json_utils.py`, `src/phytoepi/schema.py` |
| EPOP evaluation | `src/eval/evaluate_epop_relations.py` |
| Public examples | `data/examples/*.jsonl` |
| EPOP paper and dataset citation | `CITATION.cff`, `docs/epop_dataset.md`, `data/epop/manifest.json`, `data/epop/citation.bib` |

## Not Redistributed

Raw EPOP documents, third-party IE corpora, RedPajama passages, PubTator/BioC
text, model checkpoints, LoRA adapters, generated predictions, and internal
cluster scripts are intentionally excluded. These assets are either licensed,
large, private, or machine-specific. The public code takes all such paths as
arguments. EPOP documents are represented through official DOI/URL metadata
because their data-use terms prohibit redistribution or unauthorized sharing.

## Paper Consistency Notes

- Default backbone is `unsloth/llama-3-8b-bnb-4bit`.
- CPT defaults use a 4:1 plant-health/general mixture, 1B token budget,
  sequence length 4096, LoRA rank 128, rsLoRA, and packing.
- IFT defaults use sequence length 8192, LoRA rank 16, 10% internal dev split,
  seed 26, and EPOP development held out from IFT conversion.
- Inference defaults use 5 repeats, max 8128 new tokens, temperature 0.2,
  top-p 0.9, min-p 0.0, top-k 50, and seed 26.
- Evaluation reports valid JSON, parseable-only macro scores, end-to-end macro
  scores, and optional per-relation scores.
