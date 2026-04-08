# Inference Artifacts

Date locked: 2026-04-08

Every inference run must save a compact but complete artifact set.

## Mandatory Artifacts

- promoted artifact pointer
- inference artifact card
- runtime plan for each runtime family used
- sample request bundle for each mode
- runtime report stub
- latency and validation summary

## Naming Rule

Every artifact name should include:

- experiment id
- inference stage
- runtime family when applicable
