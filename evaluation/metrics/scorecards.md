# Scorecards

Date locked: 2026-04-08

Leopardi should use a small set of fixed scorecards instead of inventing new report tables every time.

## 1. Public Frontier Scorecard

Columns:

- model
- protocol version
- size band
- `page_overall`
- `markdown_validity`
- `latex_exact_match`
- `table_teds`
- `rotation_score`
- `wild_page_score`
- `p50_latency_ms_per_page`
- `params_total_b`
- evidence grade

## 2. Internal Promotion Scorecard

Columns:

- experiment id
- track
- protocol version
- internal difficulty tier
- parsing quality
- formulas
- tables
- handwriting
- graphics
- latency
- decision

## 3. Size-Normalized Scorecard

Columns:

- model
- size band
- primary public score
- latency
- footprint
- `LUS`

`LUS` should match the research definition in:

- `docs/research/unified-metrics.md`

## 4. Failure Slice Scorecard

Rows should include:

- formulas
- merged-cell tables
- handwriting
- rotation
- graphics
- document assembly
