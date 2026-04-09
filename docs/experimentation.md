# Leopardi Experimentation System

Date locked: 2026-04-07

This document defines how Leopardi runs many experiments without losing coherence.

## Why This Exists

The project is explicitly choosing a rapid-iteration strategy:

- many `~150M` experiments
- one primary GPU class
- strong ablation pressure
- later scale-up to `~500M`

Without an experiment system, this turns into noise very quickly.

## Research Operating Model

Leopardi should operate through stable tracks:

- `s0-core`
- `s0-repair`
- `s0-table`
- `s0-math`
- `s0-runtime`
- `s1-core`

Tracks exist so the project can say:

- what the current best checkpoint is
- what it replaced
- why it was promoted

## Experiment States

- `draft`
- `active`
- `candidate`
- `promoted`
- `frozen`
- `archived`

Only `promoted` and `frozen` experiments may be used as official Leopardi references.

## Required Stability Axes

To compare experiments honestly, these axes should stay explicit:

- dataset version
- split version
- benchmark protocol version
- config stack
- hardware tag
- decode mode

## Shared Run Contract

All major phases now follow the common operational contract in:

- `ops/`

This covers:

- local run layout
- heartbeat and control files
- logging policy
- external artifact persistence
- git-visible summary surfaces

## Current Readiness

The shared experiment control plane is ready.

The remaining implementation work is in the execution layers:

- training loops
- finetune loops
- optimization export backends
- inference supervisor
- evaluation runners

## Promotion Philosophy

Promotion should happen only when:

- the gain is real on both public and internal benchmarks
- the failure slices are reviewed
- latency is documented
- no data leakage concern is open

## Scaling Policy

### `Leopardi-S0`

The repo should assume many rapid experiments and frequent promotion changes.

### `Leopardi-S1`

The repo should assume far fewer but more expensive experiments.

The `S1` track must inherit from a frozen `S0` recipe.
