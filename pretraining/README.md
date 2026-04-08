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

## Final `S0` Position

For `Leopardi-S0 ~100M`, the pretraining recipe is now intentionally `exact-first`.

That is a deliberate response to the current frontier:

- PaddleOCR-VL-1.5 shows that a compact model can reach the top tier when the document objective is shaped correctly
- FireRed-OCR and olmOCR-style systems show that long-tail gains come later from targeted data engines and structured optimization
- small models degrade faster than `~0.9B` models when auxiliary or synthetic data arrives too early

Operational consequence:

- `P1` uses `tokenizer_v1` plus `p1_text_warmup_v1`
- `P2` uses `p2_exact_core_v1` plus `p2_structural_aux_v1`, with exact pairs dominant
- `P3` is the first stage that brings `p3_hardcases_v1` in as a first-class bundle

The repo can now materialize a pretraining run on disk, including:

- run manifest and heartbeat
- stage plan with scheduler, data mix, curriculum, and module-wise learning-rate policy
- report stub and checkpoint publication target

The missing execution layer is the actual training loop and dataloader integration against published bundles.
