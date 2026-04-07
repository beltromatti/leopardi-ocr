# Evaluation

Evaluation is separated from training so measurement rules remain stable across model generations.

- `datasets/`: benchmark-family adapters and official split handling
- `protocols/`: pinned evaluation protocol versions
- `runners/`: execution entry points for local shards, full sweeps, and latency runs
- `metrics/`: normalization and scoring code
- `tasks/`: dataset-specific task adapters
- `baselines/`: pinned baseline protocols and reference numbers
- `reports/`: immutable benchmark outputs and experiment reports

Leopardi should never report a single score without the corresponding latency budget.

This directory is the single home for both:

- what is being measured
- how it is being measured

There is no separate top-level `benchmark/` directory anymore.
