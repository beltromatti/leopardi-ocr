# Runtime Compatibility

Date locked: 2026-04-08

Optimization in Leopardi is runtime-aware.

## `TorchAO`

Best use:

- portable local low-bit probes
- `QAT` and `QLoRA`-aware export paths

Current posture:

- strong PyTorch-native path
- usable as a bridge to `vLLM`
- valuable for local validation and smaller-model iteration

## `LLM Compressor`

Best use:

- `vLLM`-centric deployment artifacts
- aggressive PTQ variants
- KV-cache and attention quantization experiments

Current posture:

- strongest open deployment path for `vLLM`
- compressed-tensors artifacts are a first-class target

## `vLLM`

Best use:

- default serving and benchmark baseline
- first-class target for optimized artifacts

## `SGLang`

Best use:

- high-performance serving path
- structured-output-heavy runtime experiments
- RL rollout backend

## Rule

Do not assume a variant that works in one runtime is automatically valid in the other.

Runtime portability must be measured, not inferred.
