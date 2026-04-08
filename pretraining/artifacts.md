# Pretraining Artifacts

Date locked: 2026-04-08

Each pretraining stage should emit a small set of mandatory artifacts.

## Required Artifacts

- stage plan and command snapshot
- checkpoint
- optimizer and scheduler state when resume matters
- stage config snapshot
- runtime snapshot
- data bundle identity
- validation summary
- failure summary

## Nice-To-Have Artifacts

- parameter-count card
- throughput and memory card
- gradient-norm summary
- per-slice sample exposure summary

## Publish Rule

A stage result is not a real Leopardi result until:

- the stage plan is archived
- the checkpoint is safely persisted
- the stage config is frozen
- the validation summary is archived
