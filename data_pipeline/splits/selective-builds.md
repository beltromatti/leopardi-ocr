# Selective Builds

Date locked: 2026-04-08

Leopardi should not rebuild the entire world for every run.

## Principle

Build only the bundles needed for the next experimental step.

## Examples

### Pretraining-first

Build:

- `tokenizer_v1`
- `p1_text_warmup_v1`
- `p2_exact_core_v1`

Profile:

- `exact_core_only`

### Specialist finetune push

Build:

- `f1_specialist_sft_v1`

Profile:

- `tables_only`
- `formulas_only`
- `handwriting_only`
- `hardcases_only`

### Repair iteration

Build:

- `f2_repair_sft_v1`

Profile:

- `repair_only`

### RL-only iteration

Build:

- `f3_rlvr_v1`

Profile:

- `rlvr_only`

## Persistence Rule

Once a bundle version is published and verified, later rented machines should consume it directly instead of rebuilding it unless:

- the bundle definition changed
- the source review changed
- the curation policy changed
