# Optimization Artifacts

Date locked: 2026-04-08

Every optimization run must save a compact but complete artifact set.

## Mandatory Artifacts

- source checkpoint pointer
- optimized checkpoint or runtime-only variant descriptor
- optimization config snapshot
- artifact card
- command plan
- calibration bundle pointer
- quality-retention summary
- latency and memory summary
- runtime compatibility notes

## Optional Artifacts

- per-layer quantization summary
- calibration sample manifest
- structured-output failure examples

## Naming Rule

Every artifact name should include:

- experiment id
- optimization stage
- variant id
- runtime target when applicable
