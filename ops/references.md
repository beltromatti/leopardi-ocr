# Operations References

Date locked: 2026-04-08

This file records the main sources behind the current operational design.

## Primary Inputs

### `vLLM`

Why used:

- establishes a strong serving and rollout baseline with modern persistent serving practices
- official docs confirm structured outputs as a first-class serving feature

Primary references:

- https://docs.vllm.ai/en/latest/features/structured_outputs/
- https://github.com/vllm-project/vllm

### `SGLang`

Why used:

- structured-output and high-performance serving posture relevant to RL and evaluation runs
- official docs and repo show modern structured-output and rollout relevance

Primary references:

- https://docs.sglang.io/
- https://github.com/sgl-project/sglang

### `verl`

Why used:

- practical open RL training framework with explicit runtime integration and resume-oriented workflows
- docs and README confirm `FSDP2`, CPU offload, and current `vLLM` and `SGLang` integration posture

Primary references:

- https://verl.readthedocs.io/en/latest/
- https://github.com/volcengine/verl

### `TorchAO`

Why used:

- practical compression and QAT path that affects how checkpoints and deployment artifacts should be handled

Primary references:

- https://docs.pytorch.org/ao/main/
- https://github.com/pytorch/ao

### Hugging Face Hub and Datasets

Why used:

- current large-folder upload and remote-first dataset workflows shape how Leopardi persists bundles and resumes work on ephemeral machines

Primary references:

- https://huggingface.co/docs/huggingface_hub/guides/upload#upload-a-large-folder
- https://huggingface.co/docs/datasets/stream

## Local Source Of Truth

The project-level operating model is distributed across:

- `experiments/`
- `evaluation/`
- `data_pipeline/`
- `pretraining/`
- `finetune/`

`ops/` is the common contract across all of them.
