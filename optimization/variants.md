# Optimization Variants

Date locked: 2026-04-08

Leopardi should track deployable variants as first-class artifacts.

## Canonical Families

### `bf16_reference`

Purpose:

- reference checkpoint for quality comparisons
- fallback for debugging

### `torchao_portable`

Purpose:

- portable low-bit serving candidates
- first quick quality-speed probes

Typical members:

- `torchao_int4_weight_only`
- `torchao_fp8_dynamic`

### `vllm_compressed`

Purpose:

- aggressive deployment candidates optimized for `vLLM`

Typical members:

- `llmcompressor_fp8_dynamic`
- `llmcompressor_w4a16_awq`

### `runtime_only`

Purpose:

- runtime-layer variants without new checkpoint weights

Typical members:

- `vllm_fp8_kv`
- `sglang_fp8_kv`

### `qat_export`

Purpose:

- compare QAT-derived exports against PTQ-derived exports

## Promotion Rule

No optimization variant should be promoted unless all are known:

- source checkpoint id
- optimization stage
- runtime target
- structured output mode
- benchmark protocol version
- quality retention versus `bf16_reference`
- latency and memory delta
