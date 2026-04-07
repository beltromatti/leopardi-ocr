# Data Splits

Date locked: 2026-04-08

This directory defines how Leopardi forms training and holdout bundles.

The split layer is where source pools become training reality.

## Split Principles

1. Split by document identity, not by random page only.
2. Keep release-facing holdouts stable.
3. Keep exact, synthetic, auxiliary, and weak pools explicitly separate.
4. Never let public evaluation test sets leak into release-facing training bundles.

## Files

- `split-policy.md`
  - the generic split rules
- `training-bundles.md`
  - named bundles aligned to pretraining and finetuning stages
- `holdout-governance.md`
  - internal holdout discipline and exclusion flow
