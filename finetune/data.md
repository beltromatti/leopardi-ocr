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

## Locked `S0` Finetune Footprint

`Leopardi-S0 ~200M` does not reuse the whole `~10.31M` pretraining family as if
finetuning were a second pretraining pass.

Locked targets:

- `sft_core_v1`: `240K`
- `f0_general_sft_v1`: `400K`
- `f1_specialist_sft_v1`: `700K`
- `sft_repair_v1`: `120K`
- `f2_repair_sft_v1`: `180K`
- `f3_rlvr_v1`: `120K` prompt packs

Locked stage draws:

- `F0`: `480K`
- `F1`: `720K`
- `F2`: `180K`
- `F3`: `120K`

This keeps finetuning compact, exact-anchor-heavy, and failure-driven.

## Finetune Bundle Plan

### `f0_general_sft_v1`

Sources:

- highest-confidence arXiv exact pairs
- highest-confidence PMC exact pairs
- full-page targets constructed from formula and table specialists only when they remain exact

Target:

- stable canonical Markdown plus LaTeX

Locked `S0` composition:

- `180K` arXiv exact pages
- `140K` PMC exact pages
- `40K` promoted exact full-page targets
- `40K` European multilingual synthetic pages from the Leopardi generator

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

Locked `S0` composition:

- `180K` UniMER-1M
- `90K` PubTables-1M
- `50K` FinTabNet family
- `10K` SciTSR
- `10K` CROHME
- `80K` MathWriting
- `60K` Im2LaTeX-100K
- `10K` IAM-line
- `5K` Bentham
- `5K` READ 2016
- `1K` FUNSD
- `2K` CORD
- `2K` SROIE
- `10K` ChartQA
- `10K` PlotQA
- `175K` synthetic hard cases from `synthetic_from_exact`

### `f2_repair_sft_v1`

Sources:

- corrupted model predictions from `F0` and `F1`
- public benchmark failures used only through non-train-test-overlapping mined patterns
- internal holdout-adjacent failure templates without holdout leakage
- malformed Markdown and malformed LaTeX cases generated from exact truth

Target:

- local block repair cheaper than full re-decode

Locked `S0` composition:

- `180K` block-local repairs from `model_failures_plus_exact_truth`
- anchor bundle `sft_repair_v1`: `120K`

### `f3_rlvr_v1`

Sources:

- prompt-response pairs derived from `f0_general_sft_v1`
- specialist prompts derived from `f1_specialist_sft_v1`
- repair prompts derived from `f2_repair_sft_v1`

Target:

- reward-optimizable generation tasks with exact reference targets and structural validators

Locked `S0` composition:

- `50K` general exact prompts
- `45K` specialist prompts
- `25K` repair prompts

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

## Cross-Machine Handoff

The intended sequence is:

1. machine A builds and publishes pretraining bundles
2. machine B downloads published upstream bundles from HF and builds `F0-F1`
   plus `o_calibration_docmix_v1` for later optimization
3. the first real model run publishes a failure manifest to HF
4. machine C downloads published upstream bundles plus that failure manifest and builds `F2-F3`

Generated artifacts should be persisted to HF, not committed into Git.
