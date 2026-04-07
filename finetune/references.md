# Finetune References

Date locked: 2026-04-08

This file records the main sources behind the current Leopardi finetune design.

## Core Finetune Inputs

### `verl`

Why used:

- open RL training framework with explicit support for `vLLM`, `SGLang`, LoRA-style RL, and post-training workflows

### `DAPO`

Why used:

- recent open RL recipe emphasizing dynamic sampling, token-level loss aggregation, and overlong control

### `TorchAO`

Why used:

- practical QAT and quantization path for preserving quality under low-bit deployment targets

Primary references:

- https://docs.pytorch.org/ao/0.16/eager_tutorials/finetuning.html
- https://github.com/pytorch/ao

### `vLLM`

Why used:

- safe rollout and serving baseline

### `SGLang`

Why used:

- strong structured-output posture and high-performance rollout path

## Local Source Of Truth

The detailed project-level strategy remains in:

- `docs/finetune.md`
- `docs/research/`
