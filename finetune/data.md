# Finetune Data

Date locked: 2026-04-08

Finetune data is a stricter subset and recomposition of the broader Leopardi data engine.

## Core Rule

Pretraining teaches capability.
Finetuning teaches exact behavior.

That means finetuning pools must be:

- cleaner
- better tagged
- more failure-driven
- more tightly aligned to product-visible errors

## Finetune Bundle Plan

### `f0_general_sft_v1`

Sources:

- highest-confidence arXiv exact pairs
- highest-confidence PMC exact pairs
- full-page targets constructed from formula and table specialists only when they remain exact

Target:

- stable canonical Markdown plus LaTeX

### `f1_specialist_sft_v1`

Sources:

- hard slices from exact corpora
- `PubTables-1M`
- `SciTSR`
- `CROHME`
- `MathWriting`
- `Im2LaTeX-100K`
- `IAM`
- `Bentham`
- `READ 2016`
- `FUNSD`
- `CORD`
- `SROIE`
- `ChartQA`
- `PlotQA`
- synthetic exact hard cases from `p3_hardcases_v1`

Target:

- tables, formulas, handwriting, rotation, forms, receipts, charts

### `f2_repair_sft_v1`

Sources:

- corrupted model predictions from `F0` and `F1`
- public benchmark failures used only through non-train-test-overlapping mined patterns
- internal holdout-adjacent failure templates without holdout leakage
- malformed Markdown and malformed LaTeX cases generated from exact truth

Target:

- local block repair cheaper than full re-decode

### `f3_rlvr_v1`

Sources:

- prompt-response pairs derived from `f0_general_sft_v1`
- specialist prompts derived from `f1_specialist_sft_v1`
- repair prompts derived from `f2_repair_sft_v1`

Target:

- reward-optimizable generation tasks with exact reference targets and structural validators

## Selective Build Policy

The pipeline must support preparing only the needed sources.

Valid examples:

- only exact arXiv plus PMC for `F0`
- only formula and table specialists for `F1`
- only repair data for `F2`
- only RLVR prompt packs for `F3`

This is mandatory for rented machines with finite disk and time budgets.

## Persistence Policy

Each finetune bundle must be publishable independently to persistent storage.

That means:

- prepare once
- publish once
- reuse many times

The first rented machine should not be expected to rebuild the same finetune bundle from scratch repeatedly.
