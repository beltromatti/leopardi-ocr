# Operations References

Date locked: 2026-04-08

This file records the main sources behind the current operational design.

## Primary Inputs

### `vLLM`

Why used:

- establishes a strong serving and rollout baseline with modern persistent serving practices

### `SGLang`

Why used:

- structured-output and high-performance serving posture relevant to RL and evaluation runs

### `verl`

Why used:

- practical open RL training framework with explicit runtime integration and resume-oriented workflows

### `TorchAO`

Why used:

- practical compression and QAT path that affects how checkpoints and deployment artifacts should be handled

## Local Source Of Truth

The project-level operating model is distributed across:

- `experiments/`
- `evaluation/`
- `data_pipeline/`
- `pretraining/`
- `finetune/`

`ops/` is the common contract across all of them.
