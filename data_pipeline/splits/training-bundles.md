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
- pages with both formulas and rotation-augmented views
- pages with tables plus nearby captions or prose explanation

## Bundle: `p2_structural_aux_v1`

Composition:

- PubLayNet
- DocLayNet
- PubTables-1M
- SciTSR
- FinTabNet family
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
- rotation-equivalent exact pairs
- structured handwritten notes and schedule-like pages

Why:

- this is where Leopardi must separate from cleaner born-digital parsers
- this is where compact models learn to preserve structure instead of only text under distortion

## Bundle: `sft_core_v1`

Composition:

- highest-confidence `gold_exact`
- selected `silver_exact`

Why:

- maximize precision of final Markdown plus LaTeX behavior

## Bundle: `f0_general_sft_v1`

Composition:

- same source family as `sft_core_v1`
- only samples that pass the strictest canonical-target gates

Why:

- create a clean exact-output tuning pool for `F0`

## Bundle: `f1_specialist_sft_v1`

Composition:

- merged-cell tables
- financial tables with deep headers and grouped columns
- dense formula pages
- handwriting pages
- rotated pages
- forms and receipts
- chart-heavy pages
- synthetic exact hard cases

Why:

- move the product-visible long tail that public parsers still mishandle

## Bundle: `sft_repair_v1`

Composition:

- failure slices from `p3_hardcases_v1`
- exact cases with known structural brittleness

Why:

- tighten the final behavior on the slices users notice most

## Bundle: `f2_repair_sft_v1`

Composition:

- local block failures from previous checkpoints
- gold block corrections
- canonical malformed-output repairs

Why:

- make repair mode cheaper than full page regeneration

## Bundle: `f3_rlvr_v1`

Composition:

- prompt packs from `f0_general_sft_v1`
- specialist prompt packs from `f1_specialist_sft_v1`
- repair prompt packs from `f2_repair_sft_v1`

Why:

- support objective-reward RL without rebuilding the entire supervised corpus
