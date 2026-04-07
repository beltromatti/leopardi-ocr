# Metric Catalog

Date locked: 2026-04-08

## Parsing Quality

- `page_overall`
  - benchmark-native page parsing score
- `document_overall`
  - document-level aggregate for internal holdouts
- `normalized_edit_similarity`
  - normalized string similarity after canonicalization
- `markdown_validity`
  - share of outputs that satisfy Leopardi canonical Markdown constraints
- `block_f1`
  - structure F1 over block types

## Structure

- `reading_order_edit`
- `table_teds`
- `table_teds_s`
- `layout_map`

## Math

- `formula_cdm`
- `latex_exact_match`
- `latex_norm_edit`
- `latex_compile_rate`

## Robustness

- `rotation_score`
- `handwriting_score`
- `wild_page_score`
- `photo_scan_score`
- `multilingual_score`

## Efficiency

- `p50_latency_ms_per_page`
- `p95_latency_ms_per_page`
- `pages_per_second`
- `ttft_ms`
- `output_tokens_per_page`

## Footprint

- `params_total_b`
- `params_active_b`
- `vram_peak_gib`
- `deployment_class`

## Required Metadata Fields

Every metric card must also record:

- `gpu_type`
- `precision_mode`
- `batch_size`
- `decode_mode`
- `input_resolution_policy`
- `protocol_version`
