# Model

Date locked: 2026-04-08

This directory is the architecture control plane for Leopardi model families.

Its job is to keep the model surface coherent across:

- `docs/architecture.md`
- `configs/model/`
- `src/leopardi/model/`
- downstream stages such as `pretraining/`, `finetune/`, `optimization/`, `inference/`, and `evaluation/`

It is not the Python implementation package.
The importable code lives in `src/leopardi/model/`.

## Why This Directory Exists

Leopardi already had:

- architecture research in `docs/architecture.md`
- executable code in `src/leopardi/model/`
- concrete presets in `configs/model/`

What was missing was the top-level operational layer that explains:

- which model families are canonical
- which config files are source of truth
- which interfaces downstream stages may rely on
- how the model is allowed to evolve during rapid research on rented `RTX 5090` machines

This directory fills that gap.

## Files

- `family.md`
  - canonical model-family definitions and responsibilities
- `presets.md`
  - concrete shipped presets, parameter budgets, and config mapping
- `interfaces.md`
  - tensor, output, and contract boundaries for other stages
- `artifacts.md`
  - what model assets are tracked in Git versus persistent artifact stores
- `rtx5090-runbook.md`
  - practical model-evolution policy for fast single-GPU iteration
- `references.md`
  - primary architecture references and internal research anchors

## Code And Config Entry Points

- `src/leopardi/model/`
  - importable implementation of `Leopardi-S0`
- `configs/model/leopardi_s0.yaml`
  - concrete preset currently used across the repo
- `docs/architecture.md`
  - long-form architecture blueprint and rationale
- `docs/roadmap.md`
  - phase ordering and scale-up path from `S0` to `S1`

## Current State

The model control plane and the `Leopardi-S0` implementation are aligned.

The current research vehicle is:

- `Leopardi-S0`
  - dense compact parser in the low-`90M` range
  - explicit layout-side memory fused into the latent/planner/writer path
  - native `MTP`-ready decoding heads for later speculative-serving experiments
  - optimized for rapid iteration on a rented `RTX 5090`

The next model family after the recipe is locked is:

- `Leopardi-S1`
  - same family, larger capacity, same external contract
