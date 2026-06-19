# EPOP Dataset And LREC Paper

Developer: Xinzhi Yao.

EPOP (Epidemiomonitoring of Plants) is a plant-health information-extraction
benchmark for phytosanitary monitoring reports. It covers named entities such
as Plant, Pest, Vector, Disease, Dissemination Pathway, species and geographic
normalization to NCBI Taxonomy and GeoNames, and binary or n-ary relations such
as Causes and Transmits.

## Paper

Use the LREC paper when citing the benchmark:

- Title: EPOP: A Benchmark Corpus for Assessing NLP Models on Structured
  Information Extraction in Plant Health
- Authors: Claire Nedellec, Marine Courtin, Xinzhi Yao, Marie Grosdidier,
  Isabelle Pieretti, Sandy Duperier, Robert Bossy
- Venue: Proceedings of the Fifteenth Language Resources and Evaluation
  Conference (LREC 2026), Palma, Mallorca, Spain, 11-16 May 2026
- Pages: 1331-1340
- DOI: https://doi.org/10.63317/4in2fpefq4pz
- LREC page: https://lrec.elra.info/lrec2026-main-103

## Official Dataset Sources

The data are hosted by Plateforme ESV on Recherche Data Gouv:

| Resource | DOI | Landing page | Public file |
| --- | --- | --- | --- |
| EPOP documents, version 2.0 | https://doi.org/10.57745/YKSEPY | https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi%3A10.57745%2FYKSEPY | `EPOP_documents.zip` |
| EPOP train/dev annotations, version 1.0 | https://doi.org/10.57745/ZDNOGF | https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi%3A10.57745%2FZDNOGF | `EPOP_annotations.zip` |
| Annotation guidelines | https://hal.science/hal-04744299 | https://hal.science/hal-04744299v1 | PDF on HAL |

The documents dataset contains 247 translated plant-health web documents:
110 train, 55 development, and 82 test documents. The annotation dataset
contains the gold-standard train and development annotations used for
information-extraction training and validation.

## Repository Files

This repository includes public metadata and format references, not the raw
EPOP text documents:

- `data/epop/README.md`: local placement instructions and official source URLs.
- `data/epop/manifest.json`: machine-readable paper, dataset, file, and split
  metadata.
- `data/epop/citation.bib`: BibTeX entries for the LREC paper, datasets, and
  annotation guidelines.
- `data/examples/`: tiny synthetic examples for smoke tests and format checks.

## Redistribution Note

The EPOP documents data-use terms permit text-mining use for scientific
research and prohibit redistribution or unauthorized sharing of the documents.
For that reason, the public GitHub repository records the official DOI,
landing-page URL, file manifest, citation, and expected local layout, but does
not commit the raw EPOP document text. When using the documents, attribute them
as: EPOP Documents, provided by INRAE and the ESV Platform.

## Expected Local Layout

After downloading from the official landing pages, arrange the files in one of
the layouts below or pass equivalent paths through command-line arguments:

```text
data/EPOP_documents/
  train/*.txt
  dev/*.txt
  test/*.txt

data/EPOP_json/
  train/*.json
  dev/*.json
```

The paper-aligned commands in the repository use the EPOP train split for IFT,
hold out the EPOP development split from IFT, run inference on the 55 EPOP
development documents, and evaluate predictions against the EPOP development
gold annotations.
