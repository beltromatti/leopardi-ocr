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
- obvious source-visible structure is preserved when available:
  - figure captions
  - simple tables
  - formulas
  - handwritten document sections, schedules, and warnings

Reject if:

- canonical target is empty after normalization
- block structure is clearly malformed
- high-salience source structure is silently dropped by canonicalization

## Gate 3: Markdown Structural Validity

Required:

- headings, lists, and fenced blocks obey Leopardi canonical constraints
- tables use canonical simple or complex-table representation
- structured handwritten notes remain structured Markdown, not flattened prose

Reject or demote if:

- structure is inconsistent in ways not justified by source ambiguity

## Gate 4: LaTeX Integrity

Required when formulas are present:

- formula boundaries are preserved
- LaTeX normalization succeeds
- rotation-equivalent views remain semantically consistent

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
- catch outputs that preserved text but lost table, caption, or formula structure

## Gate 6b: External Oracle Audit, Offline Only

Allowed:

- compare a tiny held-out sample against a strong external parser
- use the comparison only to improve Leopardi target rules and failure taxonomies

Not allowed:

- using external parser output as hidden automated training truth

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
