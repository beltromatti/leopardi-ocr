# Training Bundles

Date locked: 2026-04-08

This file defines the intended first bundle family for Leopardi.

## Bundle: `tokenizer_v1`

Composition:

- canonical Markdown from arXiv exact pairs
- canonical Markdown from PMC exact pairs
- LaTeX formulas from exact corpora and formula specialists
- canonical table blocks from table specialists

Why:

- tokenizer quality on Markdown and LaTeX is a direct small-model lever

## Bundle: `p1_text_warmup_v1`

Composition:

- text targets from `tokenizer_v1`
- no visual payload required

Why:

- teaches the writer side exact output style before multimodal pressure

## Bundle: `p2_exact_core_v1`

Composition:

- arXiv exact pairs
- PMC exact pairs

Oversample:

- dense formulas
- dense tables
- multi-column pages

## Bundle: `p2_structural_aux_v1`

Composition:

- PubLayNet
- DocLayNet
- PubTables-1M
- SciTSR
- CROHME
- MathWriting
- Im2LaTeX-100K

Why:

- inject explicit structure pressure where exact core data is still sparse

## Bundle: `p3_hardcases_v1`

Composition:

- synthetic exact hard cases derived from exact pools
- handwriting overlays
- forms and receipts
- chart-heavy pages
- photo or scan degradation variants

Why:

- this is where Leopardi must separate from cleaner born-digital parsers

## Bundle: `sft_core_v1`

Composition:

- highest-confidence `gold_exact`
- selected `silver_exact`

Why:

- maximize precision of final Markdown plus LaTeX behavior

## Bundle: `sft_repair_v1`

Composition:

- failure slices from `p3_hardcases_v1`
- exact cases with known structural brittleness

Why:

- tighten the final behavior on the slices users notice most
