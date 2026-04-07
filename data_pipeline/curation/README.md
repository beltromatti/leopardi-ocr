# Curation

Date locked: 2026-04-08

Curation decides what is allowed to influence Leopardi.

For a `~100M` parser, this is a first-order system, not a cleanup afterthought.

## Curation Goals

- maximize target correctness
- remove duplicate or near-duplicate supervision
- preserve hard cases instead of accidentally filtering them out
- track difficulty explicitly
- block benchmark leakage before bundle publication

## Output Classes

Every curated sample must end in exactly one class:

- `gold_exact`
- `silver_exact`
- `synthetic_exact`
- `trusted_aux`
- `weak_aux`
- `reject`

## Files

- `quality-gates.md`
  - required checks before a sample enters a bundle
- `dedup-and-leakage.md`
  - deduplication and public-benchmark contamination policy
- `difficulty-taxonomy.md`
  - difficulty labels and hard-case slices
- `acceptance-scorecards.md`
  - how samples and bundles are accepted, demoted, or rejected
