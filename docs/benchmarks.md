# Leopardi Benchmarks

Date locked: 2026-04-08

This document defines the benchmark protocol that Leopardi must use to make research claims that are hard to dispute.

The benchmark goal is not merely to report one high number.
It is to prove, on public tasks and fixed hardware, that Leopardi improves the accuracy-speed-footprint frontier.

## Claim Categories

Leopardi should make claims in five separate categories.

### 1. Overall Parsing Quality

Best overall structured document parsing quality.

### 2. PDF-to-Markdown Quality

Best fidelity on difficult PDF-to-Markdown extraction.

### 3. Exactness Specialists

Best formulas, tables, handwriting, and rotated text.

### 4. Efficiency

Best accuracy under a fixed size and hardware budget.

### 5. Size-Normalized Leadership

Best performance per parameter class.

This matters because the first research phase is `~100M`, not `~1B`.

## Hardware Protocol

Primary internal benchmark hardware:

- `1x NVIDIA RTX 5090`
- fixed software environment per benchmark release
- fixed precision mode documented for each run

Secondary comparison hardware:

- `1x H100` or equivalent datacenter GPU when reproducing external baselines

Every reported result must include:

- GPU
- precision
- batch size
- image resolution policy
- runtime family
- decode mode
- structured-output backend
- constrained-decoding mode if used
- variant family: `bf16_reference`, optimized checkpoint, or runtime-only variant

## Variant Rule

Leopardi should report two result classes separately:

- `reference` results from the canonical `bf16` artifact
- `optimized` results from promoted post-finetune variants produced by `optimization/`

This separation is mandatory.
Otherwise quality and deployment claims get mixed together.

## Output Protocol

All Leopardi models are evaluated on the same canonical output target:

- Leopardi Markdown Canonical Form
- LaTeX for formulas
- canonical complex-table fenced blocks

Before scoring, outputs must be normalized:

- whitespace normalization
- bullet normalization
- heading normalization
- math delimiter normalization
- table canonicalization

The goal is to remove stylistic noise while preserving structural correctness.

The control-plane implementation for this lives in:

- `evaluation/`
- `src/leopardi/evaluation/`

## North-Star Metrics

### Parsing Quality

- `page_overall`
- `document_overall`
- `normalized_edit_similarity`
- `block_f1`
- `markdown_validity`

### Structure

- `reading_order_edit`
- `table_teds`
- `table_teds_s`
- `layout_map` when layout labels exist

### Math

- `latex_exact_match`
- `latex_norm_edit`
- `latex_compile_rate`
- `formula_cdm`

### Robustness

- `rotation_score`
- `handwriting_score`
- `wild_page_score`
- `photo_scan_score`
- `multilingual_score`

### Efficiency

- `p50_latency_ms_per_page`
- `p95_latency_ms_per_page`
- `pages_per_second`
- `ttft_ms`
- `output_tokens_per_page`

### Footprint

- `params_total_b`
- `params_active_b`
- `vram_peak_gib`
- `deployment_class`

## Primary Public Benchmarks

### 1. OmniDocBench v1.5

Use for:

- overall structured page parsing
- formulas
- tables
- reading order

Why it matters:

- strongest public benchmark family for modern document parsing

### 2. Real5 / OmniDocBench wild extensions

Use for:

- photographed pages
- scan distortion
- perspective and illumination

Why it matters:

- many strong models still weaken badly outside clean PDFs

### 3. olmOCR-Bench

Use for:

- hard English PDF-to-Markdown extraction
- long-tail scientific and scan cases

Why it matters:

- currently one of the sharpest public tests for Markdown-linearization quality

### 4. IDP OCR Leaderboard-aligned OCR subsets

Use for:

- rotation
- handwriting
- OCR-only ceiling comparisons across general VLMs and APIs

Why it matters:

- useful for robustness sanity checks even though it is not full parsing

### 5. MDPBench

Use for:

- multilingual and photographed parsing

Why it matters:

- helps expose false “frontier” claims built only on dominant-script clean PDFs

## Specialist Benchmarks

### Tables

Use:

- `PubTables-1M`
- `SciTSR`
- `FinTabNet` or `FinTabNet.c` if ingestion and licensing are verified

Metrics:

- `TEDS`
- `TEDS-S`
- cell-span exactness

### Math

Use:

- `CROHME`
- `MathWriting`
- `Im2LaTeX-100K`

Metrics:

- exact LaTeX match
- normalized edit
- compile rate

### Handwriting

Use:

- `IAM`
- `Bentham`
- `READ 2016`

Metrics:

- CER
- WER
- block-level transcription quality after markdown normalization when applicable

### Forms and Receipts

Use:

- `FUNSD`
- `CORD`
- `SROIE`

Metrics:

- key-value structural extraction quality
- parse fidelity
- block detection and ordering

### Charts and Graphics

Use:

- `ChartQA`
- `PlotQA`
- internal chart-text parsing set mined from arXiv and PMC figures

Metrics:

- chart text recall
- legend and axis grouping quality
- caption alignment quality

## Internal Benchmark Suite

Public benchmarks are necessary but not sufficient.

Leopardi must maintain an internal holdout suite with fixed versions and no training overlap:

- born-digital scientific pages
- photographed documents
- handwriting-heavy pages
- complex tables with merged cells
- dense formula pages
- graphics and charts
- multi-page document assembly cases

Every internal suite item must be tagged by difficulty:

- `easy`
- `medium`
- `hard`
- `pathological`

## Competitor Set

Primary open competitors to reproduce or compare against:

- `PaddleOCR-VL-1.5`
- `HunyuanOCR`
- `dots.mocr`
- `FireRed-OCR`
- `GLM-OCR`
- `MonkeyOCR`
- `MinerU2.5`
- `OCRVerse`
- `olmOCR 2`
- `Infinity-Parser`
- `Chandra`
- `OpenDoc-0.1B`
- `UniMERNet` for formulas

Commercial references:

- `Mathpix`
- `Mistral OCR`
- `Gemini 2.5 Pro`
- `GPT-4.1` and `GPT-4o` class APIs when licensing permits

## Size Classes

Leopardi must report both absolute and size-normalized leaderboards.

### Size bands

- `tiny`: `<150M`
- `small`: `150M` to `<500M`
- `mid`: `500M` to `<2B`
- `large`: `2B+`

Why:

- `Leopardi-S0` should aim to dominate `tiny`
- `Leopardi-S1` should target absolute frontier, not just class frontier

## Benchmark Rules

### 1. No test leakage

Never train on:

- benchmark test sets
- benchmark validation sets used for final reporting
- benchmark-derived synthetic variants

unless explicitly disclosed in a separate ablation

Training splits may be used only when they are officially provided and kept strictly disjoint from the final reported split.

### 2. Fixed decode modes

For every model report:

- `fast`
- `standard`
- `hard`

must be evaluated separately

### 3. Confidence intervals

Use bootstrap confidence intervals on major benchmark aggregates.

### 4. Separate open and closed comparisons

Do not hide whether the comparison target is:

- open-source
- closed API
- vendor-authored claim

### 5. Publish failure slices

Every serious benchmark report must include:

- rotated pages
- handwriting
- merged-cell tables
- dense formulas
- graphics

## Leopardi Unified Score

Use the internal score defined in the research hub:

`LUS = 0.30 page_parsing + 0.15 robustness + 0.15 math + 0.10 tables + 0.10 reading_order + 0.10 latency + 0.10 footprint`

Interpretation:

- `Leopardi-S0` should optimize this score under the `tiny` size band
- `Leopardi-S1` should optimize it globally

## What Leopardi Must Beat

To justify the roadmap, the target ladder is:

### `Leopardi-S0`

Must beat:

- `OpenDoc-0.1B` on structured parsing
- `MonkeyOCR-pro-1.2B` on disclosed speed-per-quality efficiency where size-normalized
- strong OCR-only APIs on rotation and handwriting slices where possible

Should approach:

- `GLM-OCR`
- `PaddleOCR-VL-1.5`

### `Leopardi-S1`

Must aim to beat:

- `PaddleOCR-VL-1.5` on OmniDocBench-style parsing
- `dots.mocr` and `Chandra` on hard Markdown extraction
- `HunyuanOCR` on wild and multilingual parsing
- formula specialists on exact LaTeX

## Release Gates

No model is considered a release candidate unless it passes all of:

1. public benchmark gains over the previous best Leopardi checkpoint
2. no regression on formulas or merged-cell tables
3. `markdown_validity >= 99.5%`
4. documented `p50` and `p95` latency on `RTX 5090`
5. reproducible evaluation script and environment snapshot
