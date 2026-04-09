# Pairing And Targets

Date locked: 2026-04-08

This file defines how ingestion converts source corpora into Leopardi canonical targets.

## Target Authority Order

When building a canonical target, prefer this order:

1. source-native structured text
2. publisher or corpus-native XML or markup
3. trusted specialist annotations for subtasks
4. weak teacher labels only in explicitly weak pools

Teacher outputs are never allowed to masquerade as exact truth.

## Exact Pair Construction

### arXiv

Preferred path:

- source archive
- normalized markup extraction
- figure caption preservation
- simple-table recovery to GFM Markdown when rectangular
- complex-table recovery to Leopardi fenced `table` blocks when spans are present
- page alignment against compiled PDF
- page-level canonical Markdown plus LaTeX target

Primary purpose:

- exact scientific parsing supervision

### PMC Open Access

Preferred path:

- XML hierarchy
- section and caption extraction
- figure caption preservation
- table and formula preservation with canonical Markdown plus LaTeX mapping
- page alignment against PDF

Primary purpose:

- exact article parsing supervision outside arXiv

## Specialist Auxiliary Construction

### Tables

From:

- `PubTables-1M`
- `SciTSR`

Emit:

- GFM tables when the source table is simple and rectangular
- canonical complex-table blocks when rowspan or colspan is present
- topology metadata
- compact region-level Markdown targets instead of raw JSON when the source is annotation-only

### Formulas

From:

- `CROHME`
- `MathWriting`
- `Im2LaTeX-100K`
- formula spans mined from exact corpora

Emit:

- exact LaTeX targets
- formula difficulty tags
- rotation-equivalent exact targets for synthetic hard-case generation

### Handwriting

From:

- `IAM`
- `Bentham`
- `READ 2016`

Emit:

- text targets
- page- or line-level layout targets when available
- handwriting style tags
- structured Markdown when the source page clearly contains sections, schedules, warnings, or nested bullets

## Canonical Sample Requirements

Every emitted sample must include:

- stable `sample_id`
- stable `source_id`
- document or page identity
- target type
- canonical target string
- provenance metadata
- split eligibility metadata

## Mixed-Mode Data

Leopardi must explicitly support mixed samples that contain:

- printed text
- formulas
- tables
- handwriting overlays
- charts or diagrams

For such samples, truth must still be fully attributable to known source components.

## Canonical Form Bias

Leopardi canonical targets should prefer:

- standard Markdown headings, paragraphs, and lists
- GFM tables for simple rectangular tables
- fenced `table` blocks only when standard Markdown would lose structure
- LaTeX for math, preserving source-native matrix and display environments when possible

Leopardi canonical targets should avoid:

- flattening tables into prose
- dropping captions
- flattening structured handwriting into plain text
