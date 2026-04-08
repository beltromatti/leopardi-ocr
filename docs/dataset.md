# Leopardi Dataset Plan

Date locked: 2026-04-08

This document defines the exact data plan for Leopardi.

The operational build policy that implements this plan now lives in:

- `data_pipeline/`
- `src/leopardi/data_pipeline/`
- `configs/data/`

The data plan is built around one rule:

- no invented datasets
- no opaque mixtures
- every source must have a clear role in pretraining, finetuning, or evaluation

## Data Principles

### 1. Exact paired document supervision is the highest-value data

The most important data for Leopardi is not generic OCR crops.
It is page or document supervision where the target can be canonically converted to Markdown plus LaTeX.

### 2. Small models need cleaner data than large models

For `~100M`, curation quality matters more than raw scale.

### 3. Benchmark test sets are never training data

Public benchmark test sets are evaluation only.

### 4. Synthetic data is required, but the source corpus must stay explicit

We will synthesize data from public sources, not from undocumented prompts and ad hoc outputs.

## Canonical Target Sources

These are the two highest-priority paired sources for full parse supervision.

### 1. arXiv PDFs plus source files

Use:

- full parse supervision
- formulas
- scientific tables
- multi-column reading order
- figure captions

Why it matters:

- strongest public source of paired scientific PDF plus source text
- ideal for exact Markdown and LaTeX targets

Official source:

- arXiv bulk PDF and source access

Role:

- core pretraining corpus
- core SFT corpus
- core internal holdout source
- primary source family for `f0_general_sft_v1`

### 2. PubMed Central Open Access PDFs plus JATS XML

Use:

- born-digital article parsing
- headings and sections
- figure and table captions
- biomedical tables
- references and footnotes

Why it matters:

- strong second exact-pair corpus outside arXiv
- complementary style and domain

Official source:

- PMC Open Access subset with PDFs and XML

Role:

- core pretraining corpus
- core SFT corpus
- document-level assembly supervision
- primary source family for `f0_general_sft_v1`

## Layout Supervision

### 3. PubLayNet

Use:

- layout region supervision
- block detection and reading-order support

Role:

- auxiliary supervised pretraining

### 4. DocLayNet

Use:

- human-annotated layout supervision
- harder and cleaner layout evaluation than weakly labeled sources

Role:

- auxiliary supervised pretraining
- validation

## Table Supervision

### 5. PubTables-1M

Use:

- table detection
- structure recovery
- header/body segmentation

Role:

- table specialist supervision
- table evaluation

### 6. SciTSR

Use:

- scientific table structure reconstruction
- row and column topology

Role:

- table specialist supervision
- table evaluation

### 7. FinTabNet or FinTabNet.c

Use:

- dense financial tables
- complex headers and multi-row column grouping

Role:

- optional high-value table extension

Current ingestion stance:

- promoted to automated ingestion through a public aligned parquet release
- keep the original IBM-hosted lineage only as provenance, not as the live fetch path

## Formula Supervision

### 8. CROHME

Use:

- handwritten mathematical expressions
- formula structure and LaTeX exactness

Role:

- formula specialist training
- formula evaluation

### 9. MathWriting

Use:

- large-scale handwritten math recognition
- robustness to varied personal writing styles

Role:

- formula specialist training
- handwriting-plus-math supervision

### 10. Im2LaTeX-100K

Use:

- printed formula image-to-LaTeX
- exact-sequence math decoding

Role:

- formula specialist training

### 11. Formula spans mined from arXiv and PMC

Use:

- formula-in-context supervision
- inline and display math inside real pages

Role:

- bridge between isolated formula datasets and full document parsing

## Handwriting Supervision

### 12. IAM Handwriting Database

Use:

- line and page handwriting transcription

Role:

- handwriting specialist training
- handwriting evaluation

### 13. Bentham

Use:

- historical handwriting
- noisier manuscript conditions than IAM

Role:

- handwriting robustness extension

### 14. READ 2016

Use:

- historical handwritten document structure
- line and page organization

Role:

- handwriting-plus-layout supervision

## Forms, Receipts, and Noisy Business Documents

### 15. FUNSD

Use:

- noisy scanned forms
- key-value block structure

Role:

- fine-tuning on forms and noisy documents

### 16. CORD

Use:

- receipts
- structured business document reading order

Role:

- fine-tuning on receipts

### 17. SROIE

Use:

- receipt OCR and entity extraction

Role:

- receipt robustness

## Graphics and Chart Supervision

### 18. ChartQA

Use:

- chart text understanding
- chart element grounding

Role:

- chart specialist training

### 19. PlotQA

Use:

- plot reading
- axis and legend understanding

Role:

- chart specialist training

### 20. Figures and captions mined from arXiv and PMC

Use:

- chart and figure caption alignment
- scientific graphics text extraction

Role:

- chart and figure grounding extension

## Scanned Document Visual Pretraining

### 21. IIT-CDIP / RVL-CDIP family

Use:

- broad scanned-document domain coverage
- self-supervised or weakly supervised visual pretraining

Role:

- robustness pretraining
- document-domain visual adaptation

## Synthetic Data Sources

Synthetic data is required for a model this small, but the sources stay explicit.

### Synthetic source pool

Build synthetic pages and crops from:

- arXiv
- PMC
- PubTables-1M
- SciTSR
- IAM
- Bentham
- READ 2016
- CROHME
- MathWriting
- Im2LaTeX-100K
- ChartQA
- PlotQA

### Synthetic transforms

Apply:

- arbitrary rotation
- perspective warp
- blur
- low resolution
- contrast shifts
- JPEG artifacts
- annotations and stamps
- handwriting overlays
- formula overlays
- table perturbations
- chart crop perturbations

### Rendering tools

Use explicit public tooling such as:

- `SynthTIGER` for synthetic text rendering
- internal PDF and raster perturbation pipeline

## Multilingual Extension Data

The first `100M` loop should be English-heavy because exact public supervision is strongest there.

After the architecture is locked, extend multilingual coverage with:

- Wikimedia and Wikisource dumps for text content
- public PDF or HTML document sources by language where licensing is clear
- Noto font family for script coverage
- synthetic rendered document generation using the same canonical Markdown targets

This is phase-two data, not the first bottleneck.

## Evaluation-Only Data

The following are evaluation-only and must not enter the training pool for final reporting:

- `OmniDocBench` test sets
- `Real5` and related wild benchmark test sets
- `olmOCR-Bench`
- `IDP OCR leaderboard test inputs`
- `MDPBench` held-out evaluation sets
- internal release holdouts

## Data Splits

Leopardi should maintain four data layers.

### `pretrain_exact`

- arXiv
- PMC
- weakly supervised layout and table corpora

### `pretrain_synthetic`

- explicit synthetic variants from the approved source pool

### `sft_exact`

- highest-quality canonical parse pairs only

### `eval_holdout`

- public test sets plus internal non-overlapping holdouts

## Data Curation Rules

### Deduplication

Required dedup:

- document hash dedup
- page image perceptual dedup
- near-duplicate text dedup

### Leakage control

Required:

- benchmark split registry
- source-document lineage tracking
- synthetic-derivative provenance

### Quality filters

Drop or quarantine:

- broken PDF-source pairs
- malformed XML or LaTeX
- unreadable scans without usable target
- target conversions that fail canonicalization

## What To Use First For `Leopardi-S0`

The first `100M` loop should start with the smallest high-value core:

1. arXiv paired data
2. PMC OA paired data
3. PubLayNet
4. DocLayNet
5. PubTables-1M
6. SciTSR
7. CROHME
8. Im2LaTeX-100K
9. IAM
10. FUNSD
11. CORD
12. ChartQA
13. synthetic perturbations derived from the above

This is enough to discover the right architecture before expanding further.

## Official Source References

- arXiv bulk access and source files
- PMC Open Access subset
- PubLayNet
- DocLayNet
- PubTables-1M
- SciTSR
- CROHME
- MathWriting
- Im2LaTeX-100K
- IAM Handwriting Database
- Bentham
- READ 2016
- FUNSD
- CORD
- SROIE
- ChartQA
- PlotQA

For authoritative project-wide source mapping, keep this file aligned with:

- `docs/references.md`
- `docs/research/sources.md`
- `docs/research/sources-frontier-2026-04.md`
