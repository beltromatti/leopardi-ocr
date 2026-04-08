# Model Interfaces

Date locked: 2026-04-08

## Purpose

This file defines the contracts that downstream stages may rely on from the model layer.

It prevents accidental drift between:

- `src/leopardi/model/`
- `pretraining/`
- `finetune/`
- `optimization/`
- `inference/`
- `evaluation/`

## Input Boundary

`Leopardi-S0` currently accepts:

- page image tensor
- decoder input ids
- visual mode or visual token budget selector

The model itself does not directly ingest full documents or PDFs.
Those belong to upstream document ingestion and page rendering.

## Output Boundary

The canonical forward output is `LeopardiS0Output`.

Its stable groups are:

- `canonicalized`
  - normalized image plus lightweight page maps
- `visual_tokens`
  - adaptive page token sequence
- `structural_latents`
  - bottleneck latents used by planner and writer
- `planner`
  - ordered block planning signals
- `decoder_logits`
  - token logits for Markdown plus LaTeX generation
- `auxiliary`
  - rotation, handwriting, formula, and table supervision heads

## Downstream Dependencies

### `pretraining/`

May depend on:

- `decoder_logits`
- planner logits
- auxiliary heads
- visual mode selection

### `finetune/`

May depend on:

- all training-visible heads used in pretraining
- stable model config fields for stage recipes

### `optimization/`

May depend on:

- parameter count and preset identity
- architecture compatibility with export backends

It must not silently redefine model semantics.

### `inference/`

May depend on:

- decode budget assumptions
- planner-conditioned writing behavior
- validator/repair compatibility

### `evaluation/`

May depend on:

- model identity
- parameter band
- deployable artifact lineage

It must not reach inside the architecture to invent new hidden scoring behavior.

## Change Policy

Any backward-incompatible change to the model interface requires:

- preset review
- doc update in `model/`
- architecture update in `docs/architecture.md`
- validation that downstream configs still make sense
