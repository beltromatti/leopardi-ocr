# Difficulty Taxonomy

Date locked: 2026-04-08

Difficulty is not a scalar only.
Leopardi needs both tier labels and slice labels.

## Global Difficulty Tiers

- `easy`
- `medium`
- `hard`
- `pathological`

## Slice Tags

Every sample may carry zero or more of:

- `dense_formula`
- `complex_table`
- `merged_cells`
- `multi_column`
- `small_font`
- `mixed_orientation`
- `full_page_rotation`
- `local_rotation`
- `handwriting`
- `historical_handwriting`
- `form_document`
- `receipt_document`
- `chart_heavy`
- `diagram_heavy`
- `photo_scan`
- `low_contrast`
- `jpeg_artifact`
- `perspective_distortion`
- `header_footer_noise`
- `document_assembly_sensitive`

## Why This Matters

Small models improve fastest when the data engine can oversample failure slices precisely.

The pipeline should therefore support:

- slice-aware bundle composition
- slice-aware rejection analysis
- slice-aware ablation reports
