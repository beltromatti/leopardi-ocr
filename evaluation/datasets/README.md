# Evaluation Datasets

Date locked: 2026-04-08

This directory defines the benchmark-family view of data used in evaluation.

It does not store raw datasets.
It stores the measurement-facing registry of:

- benchmark families
- task role
- modality and difficulty
- split policy
- protocol coverage

## Design Rules

1. Evaluation data is grouped by benchmark family, not by training source.
2. One family can support multiple task bundles.
3. Public benchmark test sets are never repurposed as training data for promoted results.
4. Every family must declare what it is good at and what it is weak at.

## Files

- `registry.csv`
  - benchmark-family registry with scope and protocol coverage
- `public-bundles.md`
  - the benchmark bundles used for public claims
- `internal-holdouts.md`
  - the holdout philosophy for internal promotion and regression control
