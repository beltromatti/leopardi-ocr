# Pretraining

Date locked: 2026-04-08

This directory is the operational pretraining stack for `Leopardi-S0`.

It exists to turn the architecture and data blueprint into a real single-GPU training program for a rented `RTX 5090`.

## Pretraining Goal

Pretraining is not generic multimodal scaling.
For Leopardi it must maximize:

- parsing intelligence per parameter
- exact Markdown plus LaTeX behavior
- robustness to long-tail document conditions
- iteration speed on one GPU

## Files

- `recipe.md`
  - end-to-end recipe for the `S0` pretraining path
- `stages.md`
  - responsibilities and exit criteria for `P0` to `P3`
- `objectives.md`
  - loss design and auxiliary supervision policy
- `rtx5090-runbook.md`
  - realistic runtime assumptions for the rented-machine first phase
- `artifacts.md`
  - what each stage must save and publish
- `curriculum/README.md`
  - stage ordering and hard-case escalation logic

## Code And Config Entry Points

- `src/leopardi/model/`
  - model architecture implementation
- `src/leopardi/pretraining/`
  - stage config and loss-report implementation
- `configs/model/leopardi_s0.yaml`
  - concrete model config
- `configs/pretraining/`
  - stage configs
- `configs/runtime/train_rtx5090.yaml`
  - runtime defaults for the first training vehicle
- `ops/`
  - shared run, logging, control, and persistence contract

## Current State

The control plane and loss-level scaffolding are ready.

The missing execution layer is the actual training loop and dataloader integration against published bundles.
