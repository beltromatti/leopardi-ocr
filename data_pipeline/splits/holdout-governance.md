# Holdout Governance

Date locked: 2026-04-08

This file governs non-public holdouts used for Leopardi development.

## Holdout Goals

- catch overfitting to public benchmarks
- represent product-realistic document mixtures
- preserve long-tail slices that public sets underweight

## Required Holdout Buckets

- `born_digital_core`
- `photo_scan_wild`
- `handwriting_layout`
- `dense_formula`
- `complex_tables`
- `graphics_and_charts`
- `document_assembly`

## Governance Rules

1. Holdout identities are frozen once used for release gating.
2. Holdout samples may not be recycled into training bundles.
3. Any holdout refresh must create a new version, not mutate an old one.
4. Leakage review against public evaluation and training bundles is mandatory.
