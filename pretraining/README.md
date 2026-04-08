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

- `model/`
  - model-family control plane and preset policy
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

The control plane, curriculum config, optimizer grouping, loss-level implementation, and run materialization layer are ready.

The current pretraining surface already assumes:

- layout-side tokens derived from canonicalizer maps are part of the trainable memory path
- `MTP` loss is part of the compact-decoder recipe, not a late add-on

The repo can now materialize a pretraining run on disk, including:

- run manifest and heartbeat
- stage plan with scheduler, data mix, curriculum, and module-wise learning-rate policy
- report stub and checkpoint publication target

The missing execution layer is the actual training loop and dataloader integration against published bundles.
