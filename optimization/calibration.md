# Calibration Policy

Date locked: 2026-04-08

Calibration data is not ordinary training data.
It must be narrow, representative, and leakage-safe.

## Calibration Bundle

Default bundle:

- `o_calibration_docmix_v1`

It should contain:

- medium and hard pages from public training-eligible sources
- formulas, complex tables, handwriting, and chart-heavy pages
- no evaluation holdouts

## Size Rule

Use a bounded calibration set.

For `Leopardi-S0`, the first serious range is:

- `256` to `768` pages

Larger calibration sets are allowed only if they move the result materially.

## Why This Matters

If calibration is too small:

- quantized variants become noisy

If calibration is too large:

- optimization becomes slow
- the rented-machine iteration loop degrades

## Leakage Rule

Calibration pages must obey the same exclusion rules as training bundles with respect to:

- public benchmark test sets
- internal holdouts
- promotion-sensitive failure slices reserved for evaluation
