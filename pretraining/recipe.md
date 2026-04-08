# S0 Pretraining Recipe

Date locked: 2026-04-08

This is the first serious recipe for `Leopardi-S0`.

## Recipe Summary

1. train the tokenizer on canonical Markdown plus LaTeX targets
2. warm up the writer on text-only canonical targets
3. train the full model on exact page-to-Markdown parsing
4. inject explicit structure pressure through auxiliary sources
5. harden the model on synthetic and long-tail slices
6. use module-wise learning rates and token-level structure emphasis so the small model spends capacity where the frontier actually breaks

## Stage Order

### `P0`

Tokenizer only.

Data:

- `tokenizer_v1`

Goal:

- preserve Markdown, LaTeX, and table syntax efficiently

### `P1`

Text-only domain warmup for the writer path.

Data:

- `p1_text_warmup_v1`

Goal:

- strong target-language prior before multimodal decoding

### `P2`

Full multimodal core parsing stage.

Data:

- `p2_exact_core_v1`
- `p2_structural_aux_v1`

Goal:

- build the core page-to-canonical-target transduction behavior
- establish stable curriculum, sample weighting, and module-LR behavior before harder stages

### `P3`

Hard-case and robustness stage.

Data:

- `p3_hardcases_v1`

Goal:

- improve robustness without destroying the clean-core parse
- mine and replay hard failures often enough that formulas, tables, rotation, and handwriting do not regress

## Recipe Discipline

- do not skip `P1`
- do not start with the hardest pages
- do not let weak labels contaminate exact stages
- do not use identical learning rates across the full backbone once `S0` is stable enough to benefit from asymmetric tuning
- do not treat one long run as the goal; treat reproducible iteration as the goal
