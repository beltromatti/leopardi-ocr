# Output Normalization

Date locked: 2026-04-08

Normalization exists to remove formatting noise while preserving true errors.

## Allowed Normalization

- trim trailing whitespace
- normalize multiple blank lines
- normalize heading marker spacing
- normalize bullet marker spacing
- normalize inline and display math delimiters
- normalize table fence metadata ordering when semantically identical

## Forbidden Normalization

- reordering blocks
- dropping invalid blocks
- fixing malformed LaTeX silently
- collapsing merged-cell table structure into a simpler table
- inventing missing Markdown markers

## Canonical Targets

Normalization must preserve the contract defined in:

- `docs/architecture.md`
- `docs/benchmarks.md`

Especially:

- ATX headings
- canonical list forms
- `$...$` and `$$...$$`
- Markdown-native complex table fenced blocks
