# Mode Matrix

Date locked: 2026-04-08

Every serious Leopardi checkpoint must be runnable in:

- `fast`
- `standard`
- `hard`

## Required Support

### Public protocols

- `public_frontier_v1`
  - `fast`, `standard`, `hard`

### Internal holdout

- `internal_holdout_v1`
  - `standard`, `hard`

### Competitor reproduction

- `competitor_reproduction_v1`
  - use the closest reproducible mode to the competitor's official setting

## Rule

If a mode is unsupported for a model or competitor, mark it missing.
Do not invent an approximate number.
