# Experiment Registry

This directory stores the authoritative bookkeeping for all Leopardi experiments.

Files here should stay lightweight, human-readable, and append-only when practical.

## Required Files

- `experiment-index.csv`: one row per experiment
- `promotion-log.md`: promotion and demotion history
- `frozen-references.md`: locked reference checkpoints and benchmark snapshots

## Registry Rules

1. One experiment id, one row.
2. Never reuse ids.
3. Promotion is logged separately from raw experiment creation.
4. Frozen references are immutable except for metadata corrections.
