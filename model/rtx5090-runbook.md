# RTX 5090 Model Runbook

Date locked: 2026-04-08

## Purpose

This file defines how the model itself should evolve while we are iterating on rented ephemeral `RTX 5090` machines.

## Primary Rule

Prefer changes that increase research velocity and interpretability before changes that only look sophisticated on paper.

For `Leopardi-S0`, that means:

- dense over routed
- compact over bloated
- measurable over speculative
- architecture changes that preserve exportability and evaluation discipline

## Good `S0` Experiments

- latent-count sweeps
- planner-depth sweeps
- writer-depth sweeps
- visual token budget sweeps
- canonicalizer refinements that save model capacity
- auxiliary-head changes that improve training signal

## Bad `S0` Experiments

- introducing MoE before dense baselines are exhausted
- changing too many major axes in one run
- architecture changes that break `optimization/` and `inference/` compatibility
- changes that make the model impossible to retrain quickly on one GPU

## Promotion Rule

An `S0` architecture change is worth keeping only if it improves at least one of:

- parsing quality
- robustness on hard slices
- deployable latency-memory frontier
- stability of training and finetuning

without making the rest of the pipeline incoherent.

## Transition To `S1`

Do not start `S1` because `S0` is merely decent.

Start `S1` only after:

- `S0` has a stable recipe
- the evaluation protocol is producing trustworthy comparisons
- optimization and inference behavior are understood on promoted `S0` artifacts
