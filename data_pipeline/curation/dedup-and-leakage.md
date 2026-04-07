# Dedup And Leakage Control

Date locked: 2026-04-08

Leopardi must assume that public document corpora overlap in messy ways.

This file defines how to control that.

## Deduplication Layers

Use all of the following, not just one.

### 1. Source Identity Dedup

Keys:

- DOI
- arXiv id
- PMC id
- document title
- source URL or archive id

Use:

- remove exact document duplicates across mirrored sources

### 2. Canonical Text Dedup

Keys:

- normalized target hash
- MinHash or SimHash over canonical target

Use:

- remove exact or near-duplicate page targets

### 3. Visual Dedup

Keys:

- render hash
- perceptual hash

Use:

- catch raster duplicates and re-encodes

### 4. Formula And Table Dedup

Keys:

- normalized LaTeX hash
- table topology hash

Use:

- prevent specialist pools from being flooded by repeated easy patterns

## Leakage Control

`evaluation/` is the source of truth for benchmark boundaries.

Every build that targets release-facing training must exclude overlap with:

- all public benchmark test sets in `evaluation/datasets/registry.csv`
- all Leopardi internal holdout identities

## Leakage Matching Strategy

Use progressively stronger checks:

1. exact identifier match
2. normalized title match
3. canonical target near-duplicate check
4. visual near-duplicate check
5. page-neighborhood or document-family review when ambiguity remains

## Weak-Teacher Leakage Rule

If a weak teacher was run on a benchmark page for evaluation or research comparison, that output must never be recycled as training data.

## Resulting Actions

- `keep`
- `keep_but_demote`
- `hold_for_review`
- `exclude`

No ambiguous overlap should be auto-promoted to `gold_exact`.
