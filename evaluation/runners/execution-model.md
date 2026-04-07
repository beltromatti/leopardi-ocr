# Execution Model

Date locked: 2026-04-08

## Separation Of Concerns

Future runner implementations should stay separated into four conceptual layers:

1. `sample_loading`
   - consume a protocol and dataset family
2. `model_execution`
   - run the target checkpoint or baseline
3. `normalization_and_scoring`
   - apply canonical normalization and metric calculation
4. `report_materialization`
   - write immutable artifacts under `evaluation/reports/`
5. `run_contract_materialization`
   - emit manifest, heartbeat, events, and summary under the shared `ops/` contract

## Why This Matters

This separation prevents:

- hidden metric logic in inference code
- inconsistent normalization across datasets
- benchmark-specific hacks that cannot be audited
- evaluation runs that cannot be resumed or inspected cleanly over SSH
