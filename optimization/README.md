# Optimization

Date locked: 2026-04-08

This directory is the post-finetune optimization stack for `Leopardi-S0`.

Its job is to turn a strong finetuned checkpoint into deployable variants that preserve structured parsing quality while improving latency and memory use.

## Why This Stage Exists

Leopardi cannot claim frontier status from a single `bf16` checkpoint alone.

The project must know:

- which compressed variants actually preserve Markdown and LaTeX exactness
- which runtime-specific exports work best in `vLLM` and `SGLang`
- where the best quality-speed-memory frontier really is

This stage therefore sits between `finetune/` and future production `inference/`.

## Files

- `recipe.md`
  - optimization sequence from reference export to runtime-tuned variants
- `variants.md`
  - canonical deployable variant families and promotion rules
- `calibration.md`
  - how to build the calibration set without contaminating evaluation
- `runtime-compat.md`
  - `vLLM`, `SGLang`, `TorchAO`, and `LLM Compressor` compatibility matrix
- `artifacts.md`
  - required saved outputs and naming conventions
- `rtx5090-runbook.md`
  - realistic single-GPU optimization policy
- `references.md`
  - primary references behind the optimization design

## Code And Config Entry Points

- `src/leopardi/optimization/`
  - optimization config, planning, and selection logic
- `configs/optimization/`
  - stage configs `O0` to `O4`
- `configs/runtime/optimization_rtx5090.yaml`
  - runtime defaults for rented single-GPU optimization work
- `ops/`
  - shared run, logging, control, and persistence contract

## Current State

The control plane, variant recipes, runtime planning, artifact cards, and ranking logic are ready.

The repo can now materialize an optimization run on disk, including:

- run manifest and heartbeat
- per-variant artifact cards
- per-variant command plans anchored to current `TorchAO`, `LLM Compressor`, `vLLM`, and `SGLang` practice
- report stubs for later measurement

The remaining implementation step is the live backend execution wrapper that will invoke these plans automatically on a GPU-backed machine.
