# Data Pipeline

Date locked: 2026-04-08

This directory defines the full data system for Leopardi.

Its job is not just to collect datasets.
Its job is to turn heterogeneous public sources into:

- exact training pairs for Markdown plus LaTeX parsing
- hard-case synthetic data with preserved truth
- auxiliary supervision for layout, tables, formulas, handwriting, and graphics
- versioned training bundles aligned with `docs/pretrain.md`, `docs/finetune.md`, and `evaluation/`

## Core Design Rule

Leopardi is a small-model-first project.
That means data quality, target exactness, and mixture discipline matter as much as architecture.

The pipeline is therefore optimized for:

- high-value exact supervision first
- bounded local disk use on ephemeral rented machines
- persistent publication of training-ready artifacts
- explicit lineage for every promoted bundle

## Operating Model

The default operating model is:

1. acquire metadata first
2. score and select candidates
3. download only the files needed for the current build
4. transform them into canonical page or document training samples
5. publish training-ready shards and manifests to persistent storage
6. verify checksums and counts
7. purge raw transient data that is no longer needed

This is mandatory because the intended first full run happens on a rented `RTX 5090` machine with finite local storage.

## What Must Be Persistent

Persistent outside the repo:

- training-ready sample shards
- canonical manifests in machine-friendly form
- per-bundle data cards
- exclusion lists used for leakage control
- upload verification records

Persistent inside the repo:

- source registries
- bundle registries
- split definitions
- curation rules
- summary tables
- lineage notes
- publish ledgers

Raw source PDFs, source archives, and transient render caches are not repo assets.

## Storage Strategy

Leopardi should use a hybrid storage strategy.

### Primary persistent store

- Hugging Face dataset repositories

Use HF for:

- versioned sample shards
- Parquet or JSONL manifests
- data cards
- summary statistics

Why:

- strong ecosystem support
- resumable large-folder upload workflows
- straightforward distribution to training and evaluation machines

### Optional secondary mirror

- object storage such as S3, R2, or GCS

Use only as a mirror or recovery layer when the artifact volume becomes large enough to justify it.

### Git repository scope

Git stores only control-plane artifacts:

- markdown documents
- registries
- compact CSV summaries
- split policies
- publish ledgers

Never push page images, PDF corpora, or sample tar shards into this repo.

## Canonical Artifact Policy

For training, the canonical artifact is not the raw PDF.
It is a training-ready sample package that already contains:

- rendered page image or equivalent visual sample
- canonical Markdown target
- canonical LaTeX spans where applicable
- structural metadata
- provenance metadata
- split and bundle assignment

Once a source item has been converted into a verified canonical sample and published to persistent storage, the local raw copy should be eligible for deletion unless retained for an active build window.

## Data Classes

Leopardi uses four data classes.

### 1. `exact_pair`

Truth comes from source-native structure such as LaTeX or XML.

Primary examples:

- arXiv source plus compiled PDF
- PMC Open Access XML plus PDF

### 2. `synthetic_exact`

Truth is preserved under deterministic transformation or composition.

Primary examples:

- rotated or degraded exact pages
- born-digital pages augmented with handwriting overlays whose text is already known

### 3. `trusted_aux`

Auxiliary supervision for subtasks such as layout, tables, formulas, and handwriting.

Primary examples:

- PubTables-1M
- SciTSR
- CROHME
- MathWriting
- IAM

### 4. `weak_aux`

Teacher-derived or tool-derived labels used only when explicitly separated and tracked.

This class is allowed for mining and triage, but it is not allowed to silently contaminate exact bundles.

## Directory Map

- `stack.md`
  - recommended data-platform stack for real builds
- `competitive-lessons.md`
  - what current frontier systems teach us about data engines
- `references.md`
  - primary references behind the current data-engine design
- `ingestion/`
  - source registry, acquisition policy, and pairing rules
- `curation/`
  - quality gates, deduplication, difficulty tagging, and leakage control
- `synthesis/`
  - synthetic hard-case policy and transform families
- `manifests/`
  - manifest schema, artifact layout, and publish-retention policy
- `splits/`
  - split governance and named training bundles
- `registry/`
  - lightweight operational ledgers for source status, bundle status, and published assets
- `profiles/`
  - selective source and bundle build profiles for rented machines

## Alignment With The Rest Of The Repo

- `docs/dataset.md`
  - defines the strategic source plan
- `docs/pretrain.md`
  - defines stage-level training mixture goals
- `docs/finetune.md`
  - defines post-pretraining supervised data requirements
- `evaluation/`
  - defines benchmark leakage boundaries and release-facing holdouts
- `ops/`
  - defines common run layout, logging, control, and persistence policy

`data_pipeline/` turns those decisions into reproducible build policy.
