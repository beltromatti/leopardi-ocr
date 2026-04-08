# Runtime Compatibility

Date locked: 2026-04-08

Inference in Leopardi is runtime-aware by design.

## `vLLM`

Best use:

- primary serving baseline
- main production and benchmark runtime
- promoted optimized artifacts from `optimization/`

## `SGLang`

Best use:

- high-performance structured-output serving
- repair-heavy decode policies
- runtime comparison against `vLLM`

## `FlashInfer`

Best use:

- kernel layer underneath serving engines
- later custom work on batching, sampling, and attention if needed

## `xgrammar`

Best use:

- default structured generation backend
- portable constrained decoding across `vLLM` and `SGLang`

## `llguidance`

Best use:

- backend comparison on repair-heavy or schema-like tasks
- latency-sensitive grammar experiments

## Rule

Portability, latency, and structured-output exactness must all be measured.

Do not assume a runtime or grammar backend is interchangeable without evidence.
