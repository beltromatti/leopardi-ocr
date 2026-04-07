# Efficient Training Frontier

Date locked: 2026-04-07

This document focuses on how the broader frontier is improving capability per unit of training compute.

Source registry for this pass: `sources-frontier-2026-04.md`

## Main Research Themes

### 1. Better Data Beats More Data

The strongest signal here is `JEST`, which argues that batch composition and joint example selection can accelerate multimodal learning far beyond naive uniform sampling.

Additional strong open-data signal:

- `Molmo and PixMo`

Practical implications:

- quality-aware selection matters
- curriculum and hard-negative structure matter
- data engines are now a major competitive moat

For Leopardi:

- document data should be selected and mixed deliberately, not just accumulated

### 2. Synthetic Data Is Now Core Infrastructure

Best current patterns:

- `Visual Program Distillation`
- `olmOCR` synthetic benchmark generation
- `Infinity-Parser` synthetic scanned document data
- `FireRed-OCR` geometry + semantics data factory
- `UI-Venus` data cleaning and self-evolving trajectories

Practical implications:

- synthetic data is no longer just augmentation
- it is being used to create rewardable tasks and curriculum stages

For Leopardi:

- synthetic document tasks should include rotation, layout corruption, table edge cases, handwriting overlays, math perturbations, and chart-grounded structure

### 3. RL With Verifiable Rewards Is Replacing Fuzzy Preference-Only Alignment

Important signals:

- `DeepSeek-R1`
- `HybridFlow` / `verl`
- `GRPO`, `DAPO`, `DCPO`, `DaGRPO`
- `DeepVideo-R1`
- `FireRed-OCR`
- `olmOCR 2`
- `Infinity-Parser`

What changed:

- RL is increasingly tied to objective rewards
- post-training is becoming more algorithmically diverse
- throughput and stability of RL infrastructure matter almost as much as the reward

For Leopardi:

- structural rewards are a natural fit
- Markdown validity, LaTeX validity, reading order, and table structure can all be at least partly verified automatically

### 4. Distillation Is Becoming More Selective

Important signals:

- `LinguDistill`
- `Visual Program Distillation`
- reasoning distillation in `DeepSeek-V3`
- pruning-and-distillation recipes in `Minitron`

What changed:

- the frontier is moving away from blunt imitation
- selective distillation is used to recover specific capabilities or transfer verified reasoning patterns

For Leopardi:

- distill exactness and structural behavior, not just generic outputs
- preserve core language ability while specializing on documents

### 5. Training Systems Are Becoming Modular and Runtime-Aware

Important infrastructure:

- `verl`
- `TRL`
- `OpenRLHF`
- `LLaMA-Factory`
- `torchtitan`
- `FSDP2`
- `Liger-Kernel`
- `TorchAO`
- `fsdp_qlora`

Practical implications:

- the training stack is now composable
- post-training, quantization-aware training, and rollout generation are tightly coupled to serving frameworks

For Leopardi:

- do not design training in isolation from the eventual runtime

## Training Stack Recommendations for Leopardi

### Pretraining

- use a compact multimodal backbone with native-resolution support
- favor curated corpora and difficulty-tagged curriculum over raw scale
- include pure-text continuation to preserve language quality

### SFT

- stage targets by task family:
  - text/layout
  - tables
  - formulas
  - handwriting
  - charts
- use grammar-aware target normalization

### Post-Training

- use RL with verifiable rewards rather than generic preference labels alone
- treat invalid Markdown/LaTeX as hard failures
- use benchmark-derived synthetic tasks for dense hard cases

### Systems

- keep rollout generation compatible with `vLLM` and `SGLang`
- keep model training compatible with `FSDP2`, `TorchAO`, and Triton kernel improvements

## What The Frontier Suggests We Should Not Do

- do not rely only on larger model size
- do not rely only on SFT for structural exactness
- do not train only on clean born-digital PDFs
- do not leave distillation and language-capability preservation implicit
