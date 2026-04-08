# Leopardi Finetuning Plan

Date locked: 2026-04-08

This document defines the post-pretraining path for turning a good compact parser into a frontier parser.

The current implementation surface for this plan now lives in:

- `finetune/`
- `src/leopardi/finetune/`
- `configs/finetune/`
- `configs/runtime/finetune_rtx5090.yaml`

For Leopardi, finetuning is where exactness is won.

## Finetuning Stages

### F0. High-Quality General SFT

Train on the cleanest canonical parse pairs only.

Data:

- filtered arXiv pairs
- filtered PMC pairs
- high-quality table and formula examples converted into full parse targets

Primary bundle:

- `f0_general_sft_v1`

Goal:

- make the model consistently produce canonical Markdown plus LaTeX

### F1. Specialist SFT

Oversample the slices that most public parsers still mishandle:

- merged-cell tables
- dense formula pages
- rotated pages
- handwriting
- forms and receipts
- chart-heavy pages

Goal:

- strengthen the long tail without destroying the main distribution

Primary bundle:

- `f1_specialist_sft_v1`

### F2. Local Repair SFT

Train a repair mode using block-local corrupted predictions and gold corrections.

Inputs:

- original page context
- block descriptor
- invalid or low-quality block

Target:

- corrected canonical block

Goal:

- make local repair cheaper and more reliable than full re-decode

Primary bundle:

- `f2_repair_sft_v1`

### F3. RLVR

Use reinforcement learning with verifiable rewards after SFT is stable.

Framework target:

- `verl`

Rollout backends:

- `SGLang`
- `vLLM`

Primary bundle:

- `f3_rlvr_v1`

Current open compact parsers suggest two additional lessons:

- specialist SFT does most of the heavy lifting on quality
- RLVR is most useful as a sharpener for exactness and efficiency once syntax is already stable
- compact parsers benefit if layout-side memory and MTP heads stay active through finetuning instead of being treated as pretrain-only tricks

## Reward Design

Leopardi rewards should be mostly objective and automatically checkable.

### Core rewards

- Markdown validity
- LaTeX validity
- table structural validity
- reading-order consistency
- normalized edit similarity to target

### Specialist rewards

- formula exact match or normalized match
- merged-cell consistency
- header and footer suppression correctness
- chart text recall

### Efficiency-aware rewards

- output length regularization
- latency penalty
- repair budget penalty

Why this matters:

- the product target is exact structured parsing under latency pressure
- this is exactly the regime where RL with verifiable rewards makes sense

## Sampling Strategy

During finetuning, do not sample data uniformly.

Use difficulty-aware sampling with tagged buckets:

- `easy`
- `medium`
- `hard`
- `pathological`

Recommended rule:

- keep `easy` examples present for stability
- oversample `hard` and `pathological` examples enough to move failure slices
- keep an explicit failure replay buffer so regressions do not disappear between rounds

## Distillation Policy

Teacher distillation is allowed, but only under clear rules.

### Allowed

- distillation from stronger open models when licensing permits
- disagreement mining from closed APIs for analysis
- pseudo-labeling only when later validated against canonical rules

### Not allowed

- replacing core gold supervision with opaque teacher outputs
- mixing closed-model labels into evaluation holdouts

The best small model is not built by blindly copying a larger one.

## Compression-Aware Finetuning

Compression cannot be postponed.

Before locking `Leopardi-S0`, run:

- FP8 ablations if stable
- weight-only low-bit serving ablations
- QAT experiments using `TorchAO`

Also keep finetuning itself compression-aware through:

- module-wise learning-rate scaling
- KL anchoring in repair and RL stages
- reward terms that punish unnecessary length and latency
- light MTP retention so optimized serving variants do not lose their draftability

Why:

- the first architecture should already know whether it survives compression
- but the deployable artifact selection and export policy belong to the dedicated `optimization/` stage, not to finetune itself

## Failure-Driven Finetuning

Every finetune round should include mined failure cases from:

- public benchmark runs
- internal holdouts
- invalid Markdown outputs
- LaTeX syntax failures
- complex table failures
- hallucinated headers and footers

This mined set is one of the most valuable assets in the project.

## Exit Criteria For `Leopardi-S0`

`Leopardi-S0` is blueprint-complete only when all are true:

1. strong general SFT is stable
2. specialist slices no longer collapse on formulas or tables
3. repair mode improves hard pages with modest latency tax
4. RLVR improves exactness without destabilizing output format
5. compressed serving variants preserve the quality ranking

Operationally the compact-model path should behave like this:

- `F0` stabilizes syntax and reading order
- `F1` moves formulas, tables, handwriting, rotation, and charts
- `F2` teaches cheap local repair
- `F3` sharpens objective validity and latency tradeoffs without undoing the SFT gains
