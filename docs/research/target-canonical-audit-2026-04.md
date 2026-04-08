# Target Canonical Audit

Date locked: 2026-04-08

This note records a small offline audit of Leopardi target formatting against a
frontier external parser on four single-page PDFs:

- formula-heavy page
- rotated formula-heavy page
- table-heavy prospectus glossary page
- handwritten structured note page

The external parser was used only as a research oracle for target-shape review.
Its outputs are not part of the automated Leopardi training pipeline.

Reference:

- https://www.infratex.io/documentation

## Findings

### 1. Simple rectangular tables should stay standard Markdown

A glossary-style table page is well served by a normal GFM table.

Implication for Leopardi:

- prefer GFM tables whenever the source table is rectangular and span-free
- reserve fenced `table` blocks for rowspan or colspan cases only

### 2. Handwritten structured pages should not be flattened into plain text

A handwritten note page with schedules, warnings, and bullet-like structure is
better represented as headings, emphasis, and lists than as raw OCR text.

Implication for Leopardi:

- keep structured Markdown for handwritten pages when the page visibly contains
  sections, bullet lists, schedules, or warnings
- do not reduce handwriting corpora to line-OCR only in finetune bundles

### 3. Formula exactness under rotation is a real failure slice

On the rotated and non-rotated math page, structure remained strong but some
matrix entries and variable references drifted.

Implication for Leopardi:

- explicitly oversample `formula + rotation`
- keep rotation-equivalent exact targets in synthetic hard-case pools
- give formula-token and rotation-aware losses more weight in `P2`, `P3`, and
  `F1`

### 4. Captions and mixed blocks matter

The most useful outputs preserved visible structure, not just text content.

Implication for Leopardi:

- exact-core canonicalization must preserve figure captions and tables
- source-to-target conversion is part of the model recipe, not a minor ingest
  detail

## Resulting Canonical Bias

Leopardi should bias toward:

- headings, lists, emphasis, and paragraphs as first-class structure
- GFM tables for simple tables
- fenced `table` blocks for merged-cell tables
- LaTeX for formulas, including matrix environments where source-native
- caption preservation in exact-core corpora

Leopardi should avoid:

- prose-only targets for pages that visibly contain structure
- dropping captions and tables during exact-pair conversion
- treating handwriting as line transcription only when the page is structurally rich
