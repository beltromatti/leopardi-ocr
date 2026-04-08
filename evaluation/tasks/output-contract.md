# Output Contract By Task

Date locked: 2026-04-08

## Canonical Forms

### Document and page parsing

Expected output:

- Leopardi canonical Markdown
- math LaTeX for formulas, embedded inside Markdown with `$...$` or `$$...$$`
- canonical complex-table fenced blocks when needed
- preserved figure captions and handwritten page structure when source-visible
- never full TeX document source, preamble commands, package declarations, or document-level wrappers

### Formula recognition

Expected output:

- math LaTeX only
- no standalone TeX document syntax

### Table structure

Expected output:

- normalized Markdown table or canonical fenced table block

### OCR robustness

Expected output:

- canonical text or canonical Markdown block, depending on benchmark family
- structured Markdown when the page visibly contains headings, lists, schedules, or warnings

### Graphics parsing

Expected output:

- canonical Markdown with caption and chart text structure preserved where possible
