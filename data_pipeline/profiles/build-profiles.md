# Build Profiles

Date locked: 2026-04-08

These profiles define what the data engine is allowed to prepare in one build wave.

## Profile: `exact_core_only`

Build:

- arXiv exact pairs
- PMC exact pairs
- tokenizer and exact core bundles

Use when:

- bootstrapping `P0`, `P1`, `P2`, or `F0`

## Profile: `tables_only`

Build:

- `PubTables-1M`
- `SciTSR`
- optional verified `FinTabNet` family

Use when:

- pushing table-heavy pretraining or `F1`

## Profile: `formulas_only`

Build:

- `CROHME`
- `MathWriting`
- `Im2LaTeX-100K`
- exact formula spans from arXiv and PMC if already available

Use when:

- pushing formula-heavy pretraining or `F1`

## Profile: `handwriting_only`

Build:

- `IAM`
- `Bentham`
- `READ 2016`

Use when:

- pushing handwriting-heavy robustness and finetune bundles

## Profile: `hardcases_only`

Build:

- synthetic exact hard cases
- forms and receipts
- charts
- photo or scan degradation sets

Use when:

- preparing `P3`, `F1`, and `F2`

## Profile: `repair_only`

Build:

- `F2` repair packs from mined model failures and exact references

Use when:

- iterating on local repair without rebuilding the world

## Profile: `rlvr_only`

Build:

- prompt packs and reward-reference packs for `F3`

Use when:

- preparing RL runs after SFT has stabilized

## Profile: `full_frontier`

Build:

- the whole approved Leopardi stack

Use when:

- creating the first persistent gold build or a rare full rebuild
