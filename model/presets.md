# Model Presets

Date locked: 2026-04-08

## Source Of Truth

The canonical preset surface lives in:

- `configs/model/leopardi_s0.yaml`
- `src/leopardi/model/config.py`
- `src/leopardi/model/leopardi_s0.py`

This file explains how those pieces should be interpreted operationally.

## Active Preset

### `Leopardi-S0`

Status:

- active research preset
- shared default across `pretraining/`, `finetune/`, `optimization/`, `inference/`, and `evaluation/`

Current shape:

- compact dense parser
- target family size: `~100M`
- concrete implementation currently in the low-`90M` range

Main internal components:

- page canonicalizer
- adaptive visual tokenizer
- structural latent bottleneck
- block planner
- writer decoder
- auxiliary heads

## Preset Ownership

When the preset changes, all of these must remain aligned:

- `configs/model/leopardi_s0.yaml`
- `src/leopardi/model/config.py`
- `src/leopardi/model/leopardi_s0.py`
- `docs/architecture.md`
- `model/`

## Safe Changes During `S0`

Allowed:

- hidden size changes within the `S0` compact budget
- planner depth and block-count changes
- latent count and latent-layer changes
- visual token budget changes
- decoder depth and max sequence length changes
- auxiliary-head refinements

Not allowed without an explicit architecture decision:

- silent output-contract changes
- turning `S0` into a routed model
- changing the family from block-planned parsing to flat autoregression
- introducing runtime-only hacks that alter measured model quality without a new preset id

## Naming Rule

Use stable names for canonical presets:

- `leopardi_s0`
- `leopardi_s1`

Use experiment ids, not preset names, for temporary ablations.
