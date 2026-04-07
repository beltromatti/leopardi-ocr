# Split Policy

Date locked: 2026-04-08

This file defines how source pools are split before training.

## Split Axes

Use all of:

- document identity
- source family
- difficulty tier
- slice tags
- language or domain where relevant

## Mandatory Split Families

### `tokenizer_v1`

Purpose:

- tokenizer training and vocabulary diagnostics

### `p1_text_warmup_v1`

Purpose:

- text-only domain warmup

### `p2_exact_core_v1`

Purpose:

- exact multimodal core pretraining

### `p2_structural_aux_v1`

Purpose:

- layout, table, and formula auxiliary pressure

### `p3_hardcases_v1`

Purpose:

- robustness and long-tail pressure

### `sft_core_v1`

Purpose:

- high-quality exact supervised finetuning

### `sft_repair_v1`

Purpose:

- hard outputs, repair cases, and structured exactness polishing

### `internal_holdout_v1`

Purpose:

- promotion and regression control

## Split Rule

No document family may appear in both:

- release-facing training bundles
- release-facing holdout bundles

unless the holdout is from a separately curated identity space with documented non-overlap.
