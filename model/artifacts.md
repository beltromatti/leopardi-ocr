# Model Artifacts

Date locked: 2026-04-08

## Goal

Define what belongs to the model layer as durable architecture state.

## Tracked In Git

Git should keep only compact architecture source-of-truth assets:

- model-family docs in `model/`
- executable model code in `src/leopardi/model/`
- preset configs in `configs/model/`
- architecture blueprint in `docs/architecture.md`

## Not Tracked In Git

Git should not store heavyweight model outputs:

- checkpoints
- optimizer states
- exported optimized variants
- calibration tensors
- live profiling dumps

Those belong in persistent external artifact stores such as:

- Hugging Face model repositories
- checkpoint storage targets referenced in run manifests

## Minimal Durable Identity

Every serious model artifact must always retain:

- model family
- preset id
- experiment id
- parent checkpoint lineage
- parameter count
- precision family
- compatible runtime families

## Relationship To Later Stages

- `pretraining/` creates model checkpoints
- `finetune/` refines them
- `optimization/` turns them into deployable variants
- `inference/` consumes deployable variants
- `evaluation/` compares them under pinned protocols

This directory only governs the architecture-side identity of those artifacts.
