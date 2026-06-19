# EPOP Dataset Metadata

Developer: Xinzhi Yao.

This directory tracks public EPOP citation and dataset metadata for the
PhytoEpi-LLaMA workflow. It intentionally does not contain raw EPOP documents
or gold annotations.

## Official Sources

| Resource | DOI | Landing page | File |
| --- | --- | --- | --- |
| LREC 2026 paper | https://doi.org/10.63317/4in2fpefq4pz | https://lrec.elra.info/lrec2026-main-103 | LREC PDF |
| EPOP documents | https://doi.org/10.57745/YKSEPY | https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi%3A10.57745%2FYKSEPY | `EPOP_documents.zip` |
| EPOP train/dev annotations | https://doi.org/10.57745/ZDNOGF | https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi%3A10.57745%2FZDNOGF | `EPOP_annotations.zip` |
| Annotation guidelines | https://hal.science/hal-04744299 | https://hal.science/hal-04744299v1 | PDF |

## Included Here

- `manifest.json`: machine-readable source URLs, DOI values, file names,
  checksums, split counts, and expected local paths.
- `citation.bib`: BibTeX entries for the LREC paper, EPOP documents, EPOP
  annotations, and annotation guidelines.
- `file_manifest.tsv`: compact source-file inventory for the dataset archives.

## Local Use

Download the archives from the official landing pages, accept the applicable
terms there, and unpack them locally. The repository scripts accept arbitrary
paths, but the README examples use this conventional layout:

```text
data/EPOP_documents/
  train/*.txt
  dev/*.txt
  test/*.txt

data/EPOP_json/
  train/*.json
  dev/*.json
```

The EPOP documents data-use terms prohibit redistribution or unauthorized
sharing, so raw documents are excluded from GitHub. Cite and attribute the
official dataset when using it.
