# Protocol: release_gate_v1

Date locked: 2026-04-08

## Purpose

This protocol defines the minimum evidence required before a checkpoint is promoted to release-candidate status.

## Mandatory Conditions

1. `public_frontier_v1` has been run
2. `internal_holdout_v1` has been run
3. markdown validity gate passed
4. no major regression on formulas
5. no major regression on merged-cell tables
6. latency card on `RTX 5090` exists
7. report artifacts are archived
8. experiment registry entry is complete
9. if the claim uses an optimized artifact, the matching `bf16_reference` report also exists

## Required Report Artifacts

- summary report
- public benchmark table
- internal holdout table
- latency card
- failure-slice summary
- promotion recommendation
- optimized-vs-reference comparison card when applicable
