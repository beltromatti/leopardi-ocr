# Unified Metrics for Leopardi

## Why This Exists

OCR and document parsing are currently evaluated through incompatible benchmark families:

- page-level parsing benchmarks that expect Markdown or HTML
- OCR benchmarks that score plain text recognition
- component-level formula and table benchmarks
- vendor-authored benchmarks with custom rules
- latency claims measured on different hardware and page mixes

Leopardi needs one internal evaluation language that can compare all of them without pretending they are identical.

## Canonical Metric Groups

### 1. Page Parsing Accuracy

Primary question: how correct is the final page-level Markdown output?

- `page_overall`: benchmark-native overall page parsing score
- `text_edit`: normalized text edit distance or similarity
- `markdown_validity`: percentage of syntactically valid Markdown outputs
- `block_f1`: structural F1 over headings, paragraphs, lists, tables, figures, equations

### 2. Structure Fidelity

Primary question: does the model preserve document organization?

- `read_order_edit`: reading-order edit distance
- `table_teds`: tree edit distance similarity for tables
- `table_teds_s`: structure-only table TEDS where available
- `layout_map`: layout detection mAP if layout is explicit

### 3. Math Fidelity

Primary question: can the model emit usable LaTeX for math?

- `formula_cdm`: CDM on OmniDocBench
- `latex_exact_match`
- `latex_norm_edit`
- `exprate`: expression rate or equivalent when available

### 4. Robustness

Primary question: what happens off the happy path?

- `wild_page_score`: photographed/scanned benchmark performance
- `rotation_score`: rotated text or page performance
- `handwriting_score`: handwriting OCR performance
- `language_coverage`: number of supported languages and multilingual benchmark score

### 5. Efficiency

Primary question: what quality do we buy per unit of compute?

- `p50_latency_ms_per_page`
- `p95_latency_ms_per_page`
- `pages_per_second`
- `gpu_type`
- `batch_size`
- `input_resolution`
- `tokens_per_page` when disclosed

### 6. Model Footprint

Primary question: how expensive is the model to deploy?

- `params_total_b`
- `params_active_b` for MoE models
- `min_documented_vram_gib`
- `precision_mode`
- `pipeline_modules`

## Canonical Size Metric

Parameter count alone is not enough. Leopardi should track a footprint card:

1. `P_total`: total parameters in billions
2. `P_active`: active parameters in billions
3. `VRAM_min`: smallest documented serving footprint
4. `Modules`: number of separately deployed inference modules
5. `Page_tokens`: vision or sequence tokens per page when published

For leaderboard use, report:

- `size_primary = P_active` if MoE, else `P_total`
- `size_secondary = P_total`
- `deployment_class = single-model | dual-stage | multi-tool-pipeline`

## Evidence Grades

Not all numbers are equally trustworthy.

- `A`: benchmark maintainer or peer-reviewed paper on a public benchmark
- `B`: model paper or official repo on a public benchmark
- `C`: vendor-authored benchmark or in-house benchmark only
- `D`: marketing or anecdotal claim without reproducible protocol

Leopardi should never average `A` and `C` numbers without marking the evidence downgrade.

## Normalized Internal Scorecard

For internal planning, use a weighted score rather than a single public claim:

`LUS = 0.30 page_parsing + 0.15 robustness + 0.15 math + 0.10 tables + 0.10 reading_order + 0.10 latency + 0.10 footprint`

Notes:

- `page_parsing` should come from the strongest public benchmark available for the task family.
- `latency` must be normalized to fixed hardware and batch size before scoring.
- `footprint` should reward small active parameter count and lower deployment complexity.
- if a model lacks public data for a dimension, mark it missing rather than imputing.

## Benchmark Mapping

### OmniDocBench v1.5

Best public page-level benchmark for structured document parsing.

- good for: text, formulas, tables, reading order
- weak for: real-world photographed distortions unless paired with newer Real5 extensions

### Real5-OmniDocBench

Best public direction for in-the-wild physical distortions.

- good for: scan, skew, warping, screen-photography, illumination
- current issue: still young, public reporting is less dense than OmniDocBench

### olmOCR-Bench

Best current English PDF-to-Markdown stress benchmark.

- good for: old scans math, tables, headers and footers, multi-column, tiny text
- weak for: multilingual coverage

### IDP OCR Leaderboard

Good cross-provider OCR snapshot for handwriting and rotated text.

- good for: OCR-only comparability across APIs and VLMs
- weak for: full Markdown document parsing

### MDPBench

Important new multilingual benchmark.

- good for: non-Latin scripts and photographed documents
- current issue: leaderboard density is still emerging

## Leopardi Recommendation

Use at least four benchmark families in every serious model review:

1. OmniDocBench v1.5 for overall structured parsing
2. Real5-OmniDocBench or equivalent wild benchmark for robustness
3. olmOCR-Bench for English PDF-to-Markdown stress
4. IDP OCR and formula/table subsets for handwriting, rotation, and component specialists

