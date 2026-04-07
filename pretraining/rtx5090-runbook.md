# RTX 5090 Runbook

Date locked: 2026-04-08

This file defines the practical first-run assumptions for `Leopardi-S0`.

## Hardware Assumption

- one rented `RTX 5090`
- local storage is finite and ephemeral
- interruptions are possible

## Runtime Policy

- `bf16` first
- gradient checkpointing on
- micro-batch size conservative by default
- gradient accumulation used to reach effective batch size
- save often enough that eviction or rental interruption is survivable

## Resolution Policy

Do not train every sample at the hardest resolution.

Start with:

- mixed `144` and `192` DPI page renders
- hard stage allowed to use larger crop budgets selectively

## Throughput Policy

Prefer:

- stable, restartable runs
- clean metrics and checkpoints
- lower-variance ablations

Do not prefer:

- one maximally stretched run that cannot be iterated on quickly

## Checkpoint Policy

For rented hardware, checkpoint often enough that:

- at most a modest amount of work is lost on interruption
- every stage can be resumed or branched

Checkpoints should be mirrored to persistent storage quickly after validation.
