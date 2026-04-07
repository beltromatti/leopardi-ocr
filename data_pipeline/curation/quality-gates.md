# Quality Gates

Date locked: 2026-04-08

These are the minimum curation gates before a sample may enter a training bundle.

## Gate 1: Source Integrity

Required:

- source record exists in `ingestion/source-registry.csv`
- source access status is known
- sample lineage is complete

Reject if:

- source identity is missing
- document or page mapping is ambiguous

## Gate 2: Canonical Target Integrity

Required:

- target string exists
- target type is declared
- target normalization succeeds under Leopardi rules

Reject if:

- canonical target is empty after normalization
- block structure is clearly malformed

## Gate 3: Markdown Structural Validity

Required:

- headings, lists, and fenced blocks obey Leopardi canonical constraints
- tables use canonical simple or complex-table representation

Reject or demote if:

- structure is inconsistent in ways not justified by source ambiguity

## Gate 4: LaTeX Integrity

Required when formulas are present:

- formula boundaries are preserved
- LaTeX normalization succeeds

Promote to `gold_exact` only if:

- formula string is source-native or otherwise exact by construction

Demote to `silver_exact` or `trusted_aux` if:

- formula target is good but not fully source-native

## Gate 5: Visual Sample Integrity

Required:

- render exists
- page orientation metadata exists
- render is readable and non-corrupt

Reject if:

- page image is corrupt
- render crop omits material content without matching target policy

## Gate 6: Length And Density Sanity

Required:

- target length is plausible for the visual content
- text density and block count are within expected ranges for the source family

Purpose:

- catch empty, exploded, or truncated targets

## Gate 7: Difficulty Tagging

Required:

- assign at least one difficulty tier
- assign slice tags when the sample includes formulas, tables, handwriting, graphics, or rotation

Why:

- hard cases must be preserved intentionally, not lost in generic filtering

## Gate 8: Leakage Screening

Required:

- benchmark denylist screening completed
- exact and fuzzy overlap checks completed

No sample may be promoted into a release-facing bundle without this gate.
