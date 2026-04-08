# Model Family

Date locked: 2026-04-08

## Canonical Family Rule

Leopardi should evolve as one architecture family with multiple scales, not as unrelated experimental one-offs.

The canonical family currently contains:

- `Leopardi-S0`
  - compact research vehicle for fast iteration
- `Leopardi-S1`
  - later product-scale model after the recipe is proven

## `Leopardi-S0`

Purpose:

- maximize intelligence per parameter
- maximize ablation speed
- fit realistic single-GPU research on `RTX 5090`

Family constraints:

- dense model
- no MoE in the first research phase
- explicit page canonicalization
- adaptive visual tokenization
- structural latent bottleneck
- block planner
- Markdown-first writer decoder
- auxiliary supervision for rotation, handwriting, formulas, and tables

## `Leopardi-S1`

Purpose:

- scale the proven recipe without changing the external parsing contract

Allowed changes:

- more depth
- larger hidden size
- more latent capacity
- larger planner budget
- larger output length budget
- optional lightweight routed specialists only after `S0` has clearly justified them

## Family Invariants

Every Leopardi model family member must preserve:

- full-document product target with page as internal unit
- Markdown-first output contract
- LaTeX for formulas
- canonical complex-table representation
- compatibility with the shared `optimization/`, `inference/`, and `evaluation/` contracts

## Anti-Pattern Rule

Do not introduce a new model family just because one ablation looks promising.

A new family only becomes legitimate if it changes one of these fundamentals:

- visual tokenization strategy
- latent/planner/writer factorization
- output contract
- dense versus routed backbone assumption
