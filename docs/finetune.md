# Leopardi Finetuning Plan

Date locked: 2026-04-09

This document defines the post-pretraining path for turning `Leopardi-S0 ~200M`
(with pretrained SigLIP2 vision encoder and SmolLM2-initialized decoder) into a
frontier parser.

The current implementation surface for this plan now lives in:

- `finetune/`
- `src/leopardi/finetune/`
- `configs/finetune/`
- `configs/runtime/finetune_rtx5090.yaml`

For Leopardi, finetuning is where exactness is won.

Locked `S0` rule: finetuning stays compact.
The current target is `1.50M` total stage draws across `F0-F3`, not another
multi-million-scale replay of the `~10.31M` pretraining family.

Scaling rule for `S1 ~600M`:

- keep the same four-stage shape
- scale stage draws by roughly `3x`
- target `~1.2M` for `F0`, `~2.2M` for `F1`, `~500K` for `F2`, and `~300K` RLVR prompt packs

## Finetuning Stages

### F0. High-Quality General SFT

Train on the cleanest canonical parse pairs only.

Data:

- filtered arXiv pairs
- filtered PMC pairs
- high-quality table and formula examples converted into full parse targets

Primary bundle:

- `f0_general_sft_v1`
- exact anchor `sft_core_v1`

Locked `S0` size:

- `f0_general_sft_v1`: `400K`
- `sft_core_v1`: `240K`
- stage draws: `480K`

Published data pool consumed:

- `f0_general_sft_v1`
  - arXiv projected pages
  - PMC projected pages
  - approved exact full-page targets
  - Leopardi European multilingual generator
- `sft_core_v1`
  - promoted exact-only subset derived from approved full-page targets

Locked `S0` source allocation inside `f0_general_sft_v1`:

- `180K` arXiv
- `140K` PMC
- `40K` approved exact full-page targets
- `40K` Leopardi European multilingual synthetic pages

Goal:

- make the model consistently produce canonical Markdown plus LaTeX

### F1. Specialist SFT

Oversample the slices that most public parsers still mishandle:

- merged-cell tables
- dense formula pages
- rotated pages
- rotated formula pages
- handwriting
- structured handwriting pages with implicit schedule or list layout
- forms and receipts
- chart-heavy pages

Goal:

- strengthen the long tail without destroying the main distribution

Primary bundle:

- `f1_specialist_sft_v1`
- exact anchor `sft_core_v1`

Locked `S0` size:

- `f1_specialist_sft_v1`: `700K`
- `sft_core_v1`: `240K` exact anchor
- stage draws: `720K`

Published data pool consumed:

- PubTables-1M
- SciTSR
- FinTabNet family
- CROHME
- MathWriting
- Im2LaTeX-100K
- UniMER-1M
- IAM-line
- Bentham
- READ 2016
- FUNSD
- CORD
- SROIE
- ChartQA
- PlotQA
- `synthetic_from_exact`

Locked `S0` source allocation inside `f1_specialist_sft_v1`:

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
- `175K` synthetic hard cases

### F2. Local Repair SFT

Train a repair mode using block-local corrupted predictions and gold corrections.

Inputs:

- original page context
- block descriptor
- invalid or low-quality block
- block-local target class such as `table`, `formula`, `handwriting_section`, or `caption`

Target:

- corrected canonical block

Goal:

- make local repair cheaper and more reliable than full re-decode

Primary bundle:

- `f2_repair_sft_v1`
- repair anchor `sft_repair_v1`

Locked `S0` size:

- `f2_repair_sft_v1`: `180K`
- `sft_repair_v1`: `120K`
- stage draws: `180K`

### F3. RLVR

Use reinforcement learning with verifiable rewards after SFT is stable.

Framework target:

- `verl`

Rollout backends:

- `SGLang`
- `vLLM`

Primary bundle:

- `f3_rlvr_v1`

Locked `S0` size:

- `f3_rlvr_v1`: `120K` prompt packs
- stage draws: `120K`

Published data pool consumed:

- RL prompt packs derived from promoted `F0`, `F1`, and `F2` bundles

Current open compact parsers suggest additional lessons:

- specialist SFT does most of the heavy lifting on quality
- RLVR is most useful as a sharpener for exactness and efficiency once syntax is already stable
- compact parsers benefit if layout-side memory and MTP heads stay active through finetuning instead of being treated as pretrain-only tricks
- with pretrained backbones, finetuning converges faster — budget more iterations for F1 specialist and F3 RLVR stages
- SigLIP2 should remain fully unfrozen during all finetuning stages
- differential learning rates should persist: vision encoder at 1/10th base LR

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
- rotation-sensitive formula failures
- structured-handwriting formatting failures

This mined set is one of the most valuable assets in the project.

## Exit Criteria For `Leopardi-S0`

`Leopardi-S0 ~200M` is blueprint-complete only when all are true:

1. strong general SFT is stable
2. specialist slices no longer collapse on formulas or tables
3. repair mode improves hard pages with modest latency tax
4. RLVR improves exactness without destabilizing output format
5. compressed serving variants preserve the quality ranking
6. pretrained encoder features are preserved (no catastrophic forgetting)

Operationally the compact-model path should behave like this:

- `F0` stabilizes syntax and reading order (leverage SmolLM2 language priors)
- `F1` moves formulas, tables, handwriting, rotation, and charts
- `F2` teaches cheap local repair
- `F3` sharpens objective validity and latency tradeoffs without undoing the SFT gains

The pretrained backbone means F0 should converge faster than in the previous
from-scratch design. This freed budget should be invested in longer F1 specialist
training and more thorough F3 RLVR exploration.
