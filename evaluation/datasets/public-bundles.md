# Public Bundles

Date locked: 2026-04-08

This file defines the standard public evaluation bundles used by Leopardi.

## Bundle: `public_frontier_v1`

Use this bundle for:

- external claims
- competitor comparison
- release-candidate promotion

Includes:

- `OmniDocBench_v15`
- `Real5_OmniDocBench`
- `olmOCR_Bench`
- `MDPBench`
- `IDP_OCR_Leaderboard_Aligned`
- specialist subsets:
  - `PubTables_1M`
  - `SciTSR`
  - `CROHME`
  - `MathWriting`
  - `Im2LaTeX_100K`
  - `IAM`
  - `Bentham`
  - `READ_2016`
  - `FUNSD`
  - `CORD`
  - `SROIE`
  - `ChartQA`
  - `PlotQA`

## Bundle: `public_fast_path_v1`

Use this bundle for:

- latency-sensitive checkpoints
- speed-accuracy frontier analysis

Includes:

- `OmniDocBench_v15`
- `olmOCR_Bench`
- `IDP_OCR_Leaderboard_Aligned`
- selected specialist slices for:
  - tables
  - formulas
  - handwriting

## Bundle: `competitor_reproduction_v1`

Use this bundle for:

- reproducible baseline comparisons
- candidate-vs-competitor scorecards

Includes:

- `OmniDocBench_v15`
- `Real5_OmniDocBench`
- `olmOCR_Bench`
- `MDPBench`

Why this smaller bundle exists:

- not every competitor is openly reproducible on every specialist benchmark
- the core public stack must still remain fair and feasible
