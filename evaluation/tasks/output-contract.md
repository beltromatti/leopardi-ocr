# Output Contract By Task

Date locked: 2026-04-08

## Canonical Forms

### Document and page parsing

Expected output:

- Leopardi canonical Markdown
- LaTeX for formulas
- canonical complex-table fenced blocks when needed
- preserved figure captions and handwritten page structure when source-visible

### Formula recognition

Expected output:

- LaTeX only

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
