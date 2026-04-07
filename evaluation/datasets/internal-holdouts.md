# Internal Holdouts

Date locked: 2026-04-08

Public benchmarks are necessary but not sufficient.

Leopardi must maintain internal holdouts because:

- some public benchmarks are too narrow
- some do not match product difficulty
- some are too small to support regression control alone

## Internal Holdout Goals

- prevent benchmark overfitting
- stress long-tail failure slices
- support candidate promotion and demotion
- provide document-level product realism

## Required Internal Buckets

- `born_digital_core`
- `photo_scan_wild`
- `handwriting_layout`
- `dense_formula`
- `complex_tables`
- `graphics_and_charts`
- `document_assembly`

## Difficulty Tags

Every internal holdout item should carry:

- `easy`
- `medium`
- `hard`
- `pathological`

## Governance

- internal holdouts must be versioned
- lineage must be tracked back to source pools
- no promoted result is allowed without internal holdout review
