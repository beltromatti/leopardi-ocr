# Evaluation Runners

Date locked: 2026-04-08

This directory defines the execution contract for future evaluation implementations.

No code lives here yet.
The goal is to make future runner code structurally obvious and protocol-safe.

That means evaluation is specification-ready, not execution-ready.

## Runner Responsibilities

Runners must support:

- local shard validation
- full protocol sweeps
- latency-only smoke runs
- baseline comparison runs
- report materialization

## Files

- `execution-model.md`
  - expected runner roles and boundaries
- `mode-matrix.md`
  - which decode modes each runner must support
- `artifact-contract.md`
  - required output artifacts for every run
