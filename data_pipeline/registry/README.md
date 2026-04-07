# Data Registry

Date locked: 2026-04-08

This directory holds lightweight operational ledgers.

These files are intentionally compact and git-friendly.
They are not substitutes for full persistent manifests.

## Files

- `source-status.csv`
  - tracks which sources are approved, conditional, or research-only
- `bundle-registry.csv`
  - tracks named bundle families and their intended role
- `mixture-targets.csv`
  - target source-class mix for early Leopardi stages
- `finetune-mixture-targets.csv`
  - target mix for `F0` to `F3`
- `publish-registry.csv`
  - tracks published persistent artifacts and verification status
