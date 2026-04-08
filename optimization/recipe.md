# Optimization Recipe

Date locked: 2026-04-08

Leopardi optimization should proceed in five stages.

## O0. Reference Export

Always start by exporting the promoted `bf16` checkpoint as the canonical serving reference.

This reference is required for:

- all later quality-retention comparisons
- runtime debugging
- fallback serving

## O1. Portable TorchAO Variants

Use `TorchAO` for the first portable low-bit experiments.

Primary candidates:

- `int4` weight-only
- `fp8` dynamic activation plus weight quantization

Why first:

- fast iteration
- simpler local validation path
- useful even before runtime-specific exports are stable

## O2. vLLM-Centric Compression

Use `LLM Compressor` for the first aggressive deployment variants intended primarily for `vLLM`.

Primary candidates:

- `FP8` dynamic
- `W4A16` using AWQ or GPTQ-class pathways

Why:

- strongest current open deployment story for `vLLM`
- support for compressed-tensors artifacts and KV-cache work

## O3. Runtime KV And Decode Tuning

Only after a compressed checkpoint is quality-safe:

- add KV-cache quantization
- validate structured output behavior again
- benchmark `vLLM` and `SGLang` separately

## O4. QAT Export

If a finetune branch used QAT-aware preparation:

- convert the QAT-prepared checkpoint
- compare it against PTQ variants honestly

This stage exists to answer whether the extra training complexity buys real deployed wins for Leopardi.
