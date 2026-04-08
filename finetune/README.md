# Finetuning

Date locked: 2026-04-08

This directory is the operational finetuning stack for `Leopardi-S0`.

It turns a competent pretrained parser into a high-exactness production candidate.

## Finetuning Goal

Finetuning is where Leopardi wins:

- exact canonical Markdown
- exact LaTeX on formulas
- strong specialist behavior on tables, handwriting, rotation, receipts, and charts
- cheaper and more reliable local repair
- RL improvements that do not destroy format stability
- compression-aware behavior strong enough to survive the later `optimization/` stage

## Files

- `recipe.md`
  - end-to-end `F0` to `F3` path
- `data.md`
  - finetuning bundle plan and source policy
- `adapter-policy.md`
  - when to use full finetune versus LoRA-style adaptation
- `reward-design.md`
  - RLVR reward blueprint grounded in objective checks
- `references.md`
  - primary sources behind the current finetune design
- `rtx5090-runbook.md`
  - practical run policy for rented single-GPU execution
- `artifacts.md`
  - mandatory saved outputs and publish policy
- `sft/README.md`
  - SFT-specific operating rules
- `rl/README.md`
  - RLVR-specific operating rules

## Code And Config Entry Points

- `src/leopardi/finetune/`
  - finetune config, loss, reward, and runtime implementation
- `configs/finetune/`
  - stage configs `F0` to `F3`
- `configs/runtime/finetune_rtx5090.yaml`
  - baseline runtime for single-GPU finetuning
- `ops/`
  - shared run, logging, control, and persistence contract

## Current State

The control plane, recipes, losses, and rewards are ready.

The missing execution layer is the actual finetune loop, checkpoint loading, and bundle-connected dataloader path.

The deployable artifact selection and export logic now lives in:

- `optimization/`
- `src/leopardi/optimization/`
