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
- `upload_large_folder` support for shard-heavy pushes
- Xet-backed transfer acceleration where available
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

Published bundles should be consumable by streaming loaders so that later pretraining and finetuning runs can avoid full local materialization on rented machines.

For sources that require one-time manual approval or mirror pinning, the persistent Leopardi bundle becomes the durable handoff.
The rented training machine should never be the place where those access negotiations happen.

## Data Classes

Leopardi uses five data classes.

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

### 5. `derived_internal`

These are not external source corpora.
They are promoted internal derivatives built from already verified exact, synthetic, or evaluation-linked assets.

Primary examples:

- approved exact full-page target packs
- synthetic hard cases derived from exact samples
- mined failure replay packs
- RL prompt packs assembled from promoted SFT and repair bundles

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

## Current Implementation Surface

The operational code for this directory now lives in:

- `src/leopardi/data_pipeline/`
- `configs/data/`
- `configs/runtime/data_build_rtx5090.yaml`

The current code surface covers:

- source, bundle, profile, and publish registry loading
- profile-aware build planning
- source-wave scheduling for bounded local storage
- local cache and upload-staging layout
- publish-ledger materialization
- artifact plans aligned with HF dataset publication and later remote-first reuse
- canonical target normalization for TeX and JATS exact sources
- tar-shard writing and Parquet manifest emission
- real source workers for arXiv, PMC OA, DocLayNet, PubTables-1M, SciTSR, and selected HF parquet datasets
- strict manual-manifest import workers for sources that still require human-approved access or curated local mirrors
- bundle-level HF publication with verification and post-publish raw purge

At the current source-verification boundary, `sroie` and `fintabnet_family`
have been promoted to automated ingestion. `iam`, `bentham`, `read_2016`, and
`crohme` remain manual for rigor.

## Manual Source Contract

Sources that remain manual must be seeded under a local root passed with `--manual-source-root`.

Current manual-only external sources:

- `crohme`
- `iam`
- `bentham`
- `read_2016`

Expected layout:

- `<manual_root>/<source_id>/samples.jsonl`
- or `<manual_root>/<source_id>/samples.parquet`

Each row must contain:

- `sample_id`
- `canonical_target`
- `target_type`
- `task_family`
- optional `doc_id`
- optional `page_id`
- optional `asset_paths`
- optional `slice_tags`
- optional `metadata`
- optional `source_license`

`asset_paths` are resolved relative to `<manual_root>/<source_id>/`.
