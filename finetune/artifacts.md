# Finetune Artifacts

Date locked: 2026-04-08

Each finetune stage must save a compact but complete artifact package.

## Mandatory Artifacts

- checkpoint
- stage config snapshot
- runtime snapshot
- source bundle identities
- validation report
- failure-slice report

## `F2` Repair-Specific

- repair success summary
- local-repair latency summary
- invalid-block before and after statistics

## `F3` RL-Specific

- reward breakdown summary
- rollout runtime summary
- response-length summary
- instability notes if any

## Publish Rule

A finetune result is not real until:

- checkpoint persisted
- stage report archived
- exact bundle identities recorded
