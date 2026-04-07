# Protocol: public_frontier_v1

Date locked: 2026-04-08

## Purpose

This is the primary protocol for any public-facing Leopardi claim.

Use it for:

- candidate promotion
- public scorecards
- papers and technical reports
- competitor comparisons

## Benchmark Bundle

Use dataset bundle:

- `public_frontier_v1`

## Required Reporting Axes

- overall structured parsing
- PDF-to-Markdown quality
- formulas
- tables
- handwriting and rotation
- multilingual and photographed robustness
- latency
- footprint

## Hardware Rules

Primary hardware:

- `1x RTX 5090`

Secondary comparison hardware when reproducing competitors:

- `1x H100` or the closest feasible equivalent

Always report:

- GPU
- precision
- batch size
- decode mode
- visual resolution policy
- constrained-decoding mode if any

## Decode Modes

Run separately for:

- `fast`
- `standard`
- `hard`

Never average them into one number without preserving the per-mode card.

## Normalization Rules

Outputs must be normalized according to:

- `evaluation/metrics/normalization.md`

Required canonical aspects:

- heading normalization
- whitespace normalization
- bullet normalization
- math delimiter normalization
- table canonicalization

## Core Metrics

Use the metric catalog in:

- `evaluation/metrics/catalog.md`

And scorecards in:

- `evaluation/metrics/scorecards.md`

## Evidence Policy

When competitor numbers are included, mark evidence grade:

- `A`
- `B`
- `C`
- `D`

according to:

- `evaluation/baselines/reproduction-policy.md`
- `docs/research/unified-metrics.md`

## Mandatory Public Tables

Every report using this protocol must include:

1. public benchmark aggregate table
2. latency and footprint card
3. failure-slice summary
4. competitor comparison table with evidence grades
