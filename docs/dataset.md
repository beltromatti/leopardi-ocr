# Leopardi Dataset Plan

Date locked: 2026-04-09

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
It is page or document supervision where the target can be canonically converted to Markdown plus math LaTeX.

Here `LaTeX` means only the math notation used inside Markdown output:

- inline math: `$...$`
- display math: `$$...$$`

It does not mean full TeX document syntax with preamble, packages, theorem setup, bibliography commands, or other source-level scaffolding.

### 2. Small models need cleaner data than large models, but also need enough

For `~150M`, curation quality matters more than raw scale — but the quantity
gap with competitors must be closed. Research (SAIL-VL, Beyond Chinchilla)
shows that more high-quality data always helps, following logarithmic scaling.

Operational implication:

- exact page or document pairs stay dominant through `P2`
- specialist and synthetic sources are staged in, not dumped in uniformly from the beginning
- `F0` and `F1` both retain exact anchors during finetuning
- target total for S0: ~10.3M samples
  - ~5.31M real-source samples
  - ~5.0M synthetic samples
- target total for S1: ~30M samples
  - ~15M real-source samples
  - ~15M synthetic samples

### 3. Benchmark test sets are never training data

Public benchmark test sets are evaluation only.

### 4. Synthetic data is required, but the source corpus must stay explicit

We will synthesize data from public sources, not from undocumented prompts and ad hoc outputs.

## Evidence-Based Competitive Position

Current evidence supports the following claim:

- this data plan is strong enough to make `Leopardi-S0 ~150M` highly competitive in the compact-parser regime
- it is not scientifically defensible to promise that a `150M` model will automatically beat every `~0.9B` frontier parser on every benchmark before training and evaluation happen

The strongest reasons the plan is still worth pursuing are:

- the exact-pair core is cleaner and more Markdown-native than many competitor stacks
- table, formula, handwriting, receipt, and chart specialists are explicitly separated instead of being left implicit in generic OCR mixtures
- the pipeline now preserves a compact-model-friendly curriculum where exact supervision dominates before robustness pressure rises

The strongest remaining structural disadvantage versus a model like `PaddleOCR-VL-1.5` is multilingual breadth.
That is a real risk and must remain explicit until a stronger multilingual document source family is promoted.

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
- compact Markdown layout-region targets instead of raw annotation dumps

Role:

- auxiliary supervised pretraining

### 4. DocLayNet

Use:

- human-annotated layout supervision
- harder and cleaner layout evaluation than weakly labeled sources
- compact Markdown layout-region targets instead of raw annotation dumps

Role:

- auxiliary supervised pretraining
- validation

## Table Supervision

### 5. PubTables-1M

Use:

- table detection
- structure recovery
- header/body segmentation
- canonical table-region targets for non-rectangular specialist supervision

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

- table specialist training
- specialist SFT extension for deep financial headers and grouped columns

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

Current ingestion stance:

- automated via pinned public complete mirror aligned to CROHME train and benchmark splits

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

### 10b. UniMER-1M

Use:

- 1M+ diverse formula image-to-LaTeX pairs
- covers printed, handwritten, simple, and complex expressions
- includes arXiv formulas, Pix2tex, CROHME, and HME100K subsets

Why it matters:

- 10× larger than Im2LaTeX-100K
- much more diverse in expression complexity and style
- Apache 2.0 license, verified accessible on HuggingFace
- developed by OpenDataLab alongside the CDM evaluation metric

Official source:

- `wanderkid/UniMER_Dataset` on HuggingFace

Role:

- primary formula specialist training source for S0 and S1
- bridges the gap between isolated formula datasets and the quantity
  needed for robust formula recognition at scale

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

Current ingestion stance:

- automated via the `Teklia/IAM-line` mirror with clean train/validation/test line-level splits; official IAM archives remain the provenance reference but are still login-gated

### 13. Bentham

Use:

- historical handwriting
- noisier manuscript conditions than IAM

Role:

- handwriting robustness extension

Current ingestion stance:

- automated via official Zenodo Bentham Dataset R0 image and ground-truth archives

### 14. READ 2016

Use:

- historical handwritten document structure
- line and page organization

Role:

- handwriting-plus-layout supervision

Current ingestion stance:

- automated via official Zenodo `PublicData.tgz` with PAGE-XML page supervision

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

Current ingestion stance:

- promoted to automated ingestion through a public curated release with explicit license metadata

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

The `150M` S0 loop is English-heavy because exact public supervision is
strongest there. But minimal multilingual coverage is now included from S0
to avoid being completely excluded from multilingual benchmarks like MDPBench.

### S0 Multilingual Sources

#### SynthDoG-European (DE, FR, ES, IT, PT — generated at build time)

Source: generated on the rented machine using the open-source SynthDoG tool
from the Donut project with Wikipedia dumps in each language and Noto fonts
License: generated data, Apache 2.0 (tool license)
S0 target: 20,000 pages per language = 100,000 total
Generated using: `github.com/clovaai/donut/synthdog/`

Why European languages:

- Leopardi targets academic, scientific, and business documents — a primarily
  anglophone and European market
- German, French, Spanish, Italian, Portuguese cover the major European
  scientific and legal document languages
- Latin-script languages cost almost zero to the tokenizer (shared alphabet)
- CJK would require a much larger vocabulary and more parameters to handle well
- the 150M S0 model should focus on what it can do best with limited capacity

Generation pipeline:

1. Download Wikipedia dumps for DE, FR, ES, IT, PT
2. Run SynthDoG with Noto fonts and document backgrounds
3. Canonicalize output into Leopardi sample format (image + ground_truth text)
4. Publish as a Leopardi bundle on HuggingFace

### S1 Multilingual Extension

For S1, expand multilingual coverage with:

- full SynthDoG-European: 100K per language = 500K
- EUR-Lex legal documents rendered as synthetic PDFs in 23 EU languages
- arXiv non-English papers: filtered subset with language tagging
- Optional CJK expansion once European coverage is proven

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

The `150M` S0 loop uses all high-value sources at scaled quantity:

1. arXiv paired data (`250K` source docs, target `~2.0M` projected pages)
2. PMC OA paired data (`150K` source docs, target `~1.2M` projected pages)
3. UniMER-1M (`1.0M` formula pairs)
4. PubLayNet (`300K` layout samples)
5. DocLayNet (`80,863` layout samples)
6. PubTables-1M (`250K` table samples)
7. SciTSR (`15K` table samples)
8. FinTabNet family (`100K` financial-table samples)
9. CROHME (`10K` handwritten formula samples)
10. MathWriting (`200K` handwritten math samples)
11. Im2LaTeX-100K (`100K` printed formula samples)
12. IAM-line (`10,373` handwriting line samples)
13. Bentham (`5K` page samples)
14. READ 2016 (`5K` page samples)
15. FUNSD, CORD, SROIE (`5K` forms and receipts total)
16. ChartQA, PlotQA (`30K` charts and plots total)
17. SynthDoG-European DE/FR/ES/IT/PT (`500K` generated at build time, `100K` per language)
18. Synthetic perturbations with text corruption (`~4.5M` hard cases derived from exact sources)

Target total for the published `S0` data family: `~10.31M` samples.

- `~5.31M` real-source samples
- `~500K` build-time synthetic multilingual samples
- `~4.5M` derived synthetic hard cases

Locked `S0` finetune family on top of that pretraining corpus:

- `sft_core_v1`: `240K`
- `f0_general_sft_v1`: `400K`
- `f1_specialist_sft_v1`: `700K`
- `sft_repair_v1`: `120K`
- `f2_repair_sft_v1`: `180K`
- `f3_rlvr_v1`: `120K` prompt packs

Locked `S0` finetune stage draws:

- `F0`: `480K`
- `F1`: `720K`
- `F2`: `180K`
- `F3`: `120K`

This puts `Leopardi-S0` into a genuinely frontier-scale data regime for a compact parser,
while still keeping the exact-pair core dominant and auditably cleaner than most competitor stacks.

### S0 Capacity Table

| Source family | S0 target |
| --- | ---: |
| arXiv projected pages | ~2,000,000 |
| PMC projected pages | ~1,200,000 |
| UniMER-1M | 1,000,000 |
| PubLayNet | 300,000 |
| DocLayNet | 80,863 |
| PubTables-1M | 250,000 |
| SciTSR | 15,000 |
| FinTabNet family | 100,000 |
| CROHME | 10,000 |
| MathWriting | 200,000 |
| Im2LaTeX-100K | 100,000 |
| IAM-line | 10,373 |
| Bentham | 5,000 |
| READ 2016 | 5,000 |
| FUNSD | 1,000 |
| CORD | 2,000 |
| SROIE | 2,000 |
| ChartQA | 15,000 |
| PlotQA | 15,000 |
| Real-source subtotal | ~5,311,236 |
| SynthDoG-European | 500,000 |
| synthetic_from_exact | ~4,500,000 |
| Synthetic subtotal | ~5,000,000 |
| Total S0 family | ~10,311,236 |

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
