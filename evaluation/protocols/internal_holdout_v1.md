# Protocol: internal_holdout_v1

Date locked: 2026-04-08

## Purpose

This protocol governs internal promotion and regression control.

It exists because public benchmarks alone are not enough to decide what gets promoted.

## Dataset Bundle

Use dataset family:

- `Leopardi_Internal_Holdout`

Required buckets:

- `born_digital_core`
- `photo_scan_wild`
- `handwriting_layout`
- `dense_formula`
- `complex_tables`
- `graphics_and_charts`
- `document_assembly`

## Required Difficulty Breakdown

Each result must be reported by:

- `easy`
- `medium`
- `hard`
- `pathological`

## Required Metrics

- canonical parsing quality
- markdown validity
- latex validity
- table structure quality
- document assembly quality
- p50 and p95 latency

## Promotion Role

No model may be promoted on public numbers alone if it regresses materially on:

- formulas
- merged-cell tables
- handwriting
- graphics
- document assembly
