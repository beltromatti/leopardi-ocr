# Leopardi Optimization Plan

Date locked: 2026-04-08

This document defines the post-finetune optimization plan for `Leopardi-S0`.

The current implementation surface for this plan now lives in:

- `optimization/`
- `src/leopardi/optimization/`
- `configs/optimization/`
- `configs/runtime/optimization_rtx5090.yaml`

The optimization objective is not merely to compress the model.
It is:

- preserve exact Markdown plus LaTeX behavior
- minimize deployed latency
- reduce memory enough to widen deployment options
- keep runtime behavior honest for `evaluation/`

## Stage Summary

Leopardi optimization should happen in five stages.

### O0. Reference Export

Export the canonical `bf16` serving artifact.

### O1. Portable TorchAO Variants

Probe `TorchAO`-based low-bit variants that are easy to validate locally.

### O2. vLLM-Centric Compression

Use `LLM Compressor` to create stronger `vLLM`-aligned artifacts such as `FP8` and `W4A16`.

### O3. Runtime KV Variants

Apply runtime-layer KV optimizations only after a checkpoint variant is already quality-safe.

### O4. QAT Export

If a finetune branch was QAT-aware, convert and compare it against the PTQ frontier honestly.

## Promotion Rule

An optimization variant is promotable only when all are true:

1. quality retention versus `bf16_reference` is within the allowed budget
2. Markdown and LaTeX validity stay above floor
3. latency and memory improve materially
4. runtime-specific structured output still behaves correctly
5. evaluation protocol and runtime preset are pinned

## What Not To Do

### 1. Do not optimize before the finetuned checkpoint is stable

That only obscures training problems.

### 2. Do not benchmark optimized artifacts against competitors without a `bf16` reference

Otherwise the quality-speed claim becomes fuzzy.

### 3. Do not mix runtime-only tuning with checkpoint compression silently

They are different experiment classes and must stay distinguishable.
