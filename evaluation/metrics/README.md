# Evaluation Metrics

Date locked: 2026-04-08

This directory defines the operational metric system for Leopardi.

It turns the research-layer synthesis in `docs/research/unified-metrics.md` into a stable measurement contract.

## Files

- `catalog.md`
  - metric definitions and required reporting fields
- `normalization.md`
  - canonical output normalization rules
- `scorecards.md`
  - standard tables used for model review and promotion

## Rules

1. Metrics must be comparable across time.
2. Normalization may remove stylistic noise but must not forgive structural errors.
3. Latency and footprint are part of the evaluation surface, not optional add-ons.
