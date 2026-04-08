# Inference

Date locked: 2026-04-08

This directory is the serving and decoding stack for promoted Leopardi artifacts.

Its job is to turn an optimized checkpoint into the fastest possible document parser without giving away Markdown and LaTeX exactness on hard pages.

## Why This Stage Exists

Leopardi will not win on raw checkpoint quality alone.

The project must also know:

- how to route easy versus hard pages
- when to spend extra compute and when not to
- which structured decoding backend is worth its latency
- how to keep production and evaluation on the same runtime contract

This stage therefore sits after `optimization/` and below `evaluation/`.

## Files

- `routing.md`
  - adaptive page routing and mode budgets
- `structured-decoding.md`
  - grammar, regex, and repair policy
- `validators.md`
  - exactness checks for Markdown and LaTeX
- `document-assembly.md`
  - page-to-document assembly policy
- `runtime-compat.md`
  - `vLLM`, `SGLang`, `FlashInfer`, `xgrammar`, and `llguidance`
- `artifacts.md`
  - required saved runtime plans and reports
- `rtx5090-runbook.md`
  - realistic single-GPU serving posture
- `references.md`
  - primary references behind the inference design

## Code And Config Entry Points

- `src/leopardi/inference/`
  - inference config, routing, validation, assembly, and runtime planning
- `configs/inference/`
  - stage configs for adaptive `vLLM` and structured `SGLang`
- `configs/runtime/inference_rtx5090.yaml`
  - runtime defaults for rented single-GPU inference work
- `ops/`
  - shared run, logging, control, and persistence contract

## Current State

The control plane, routing logic, validator layer, document assembly, launch planning, and artifact cards are ready.

The repo can now materialize an inference run on disk, including:

- run manifest and heartbeat
- runtime launch plans for `vLLM` and `SGLang`
- sample request payloads for `fast`, `standard`, and `hard`
- inference artifact cards and report stubs

The remaining implementation step is the live Python supervisor that will boot the chosen runtime, execute requests, and push reports automatically on a GPU-backed machine.
