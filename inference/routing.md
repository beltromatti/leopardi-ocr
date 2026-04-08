# Inference Routing

Date locked: 2026-04-08

Leopardi should not spend the same budget on every page.

## Modes

- `fast`
  - clean pages, low density, no strong specialist trigger
- `standard`
  - moderate density or uncertain structure
- `hard`
  - formulas, merged-cell tables, handwriting, charts, long tiny text, or photographed distortion

## Signals

Routing is driven by cheap pre-decode signals:

- visual density
- estimated block count
- formula density
- table density
- handwriting likelihood
- chart likelihood
- long tiny text likelihood
- photo distortion likelihood
- orientation uncertainty

## Policy

- hard specialist triggers override the scalar complexity score
- otherwise use a weighted score and fixed thresholds
- log the final route decision and reasons for every page

This is the right posture for a `~100M` model because it preserves latency on the easy majority while still spending extra compute where accuracy actually moves.
