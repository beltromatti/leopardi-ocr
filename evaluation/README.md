# Evaluation

Evaluation is separated from training so benchmark rules remain stable across model generations.

- `metrics/`: normalization and scoring code
- `tasks/`: dataset-specific task adapters
- `baselines/`: pinned baseline protocols and reference numbers

Leopardi should never report a single score without the corresponding latency budget.

