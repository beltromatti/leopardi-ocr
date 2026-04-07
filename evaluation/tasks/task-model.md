# Task Model

Date locked: 2026-04-08

## Core Task Types

### `document_parsing`

Input:

- full document or page sequence

Output:

- canonical Markdown document

Main metrics:

- `document_overall`
- `markdown_validity`
- `reading_order_edit`

### `page_parsing`

Input:

- page unit

Output:

- canonical Markdown plus LaTeX

Main metrics:

- `page_overall`
- `block_f1`
- `markdown_validity`

### `pdf_to_markdown`

Input:

- PDF page or rendered page

Output:

- flattened but canonical Markdown output

Main metrics:

- `normalized_edit_similarity`
- `markdown_validity`

### `formula_recognition`

Input:

- formula crop or formula-containing region

Output:

- LaTeX

Main metrics:

- `latex_exact_match`
- `latex_norm_edit`
- `latex_compile_rate`

### `table_structure`

Input:

- page or table region

Output:

- canonical table representation

Main metrics:

- `table_teds`
- `table_teds_s`

### `ocr_robustness`

Input:

- rotated, handwritten, or distorted page or line

Output:

- text or canonical block output depending on the benchmark

Main metrics:

- `rotation_score`
- `handwriting_score`

### `graphics_parsing`

Input:

- figure, chart, or chart-heavy page

Output:

- chart-related text and structure in canonical output

Main metrics:

- chart text recall
- caption alignment
- structure grouping quality
