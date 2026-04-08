# Optimization References

Date locked: 2026-04-08

This file records the main sources behind the current Leopardi optimization design.

## Primary Inputs

### `TorchAO`

Why used:

- PyTorch-native QAT, QLoRA, and low-bit serving path

Primary references:

- https://docs.pytorch.org/ao/stable/
- https://docs.pytorch.org/ao/0.16/eager_tutorials/finetuning.html
- https://github.com/pytorch/ao

### `LLM Compressor`

Why used:

- strongest open post-training compression path aligned to `vLLM`

Primary references:

- https://docs.vllm.ai/projects/llm-compressor/en/latest/
- https://github.com/vllm-project/llm-compressor

### `vLLM`

Why used:

- primary serving baseline and optimized artifact target

Primary references:

- https://docs.vllm.ai/en/latest/features/structured_outputs/
- https://github.com/vllm-project/vllm

### `SGLang`

Why used:

- high-performance serving and structured-output runtime target

Primary references:

- https://docs.sglang.ai/
- https://docs.sglang.ai/backend/sampling_params.html
- https://github.com/sgl-project/sglang

## Competitor Inputs

### `GLM-OCR`

Why used:

- explicit public deployment story across `vLLM`, `SGLang`, and alternative backends

### `DeepSeek-OCR`

Why used:

- clear `vLLM` integration and native-resolution serving posture

### `Chandra`

Why used:

- practical dual path between local/HF and `vLLM` serving

### `FireRed-OCR`

Why used:

- strong evidence that structured parsing quality wins only matter if the serving path preserves them
