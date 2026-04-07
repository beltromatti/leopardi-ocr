# Pretraining Stages

Date locked: 2026-04-08

This file defines what each pretraining stage is supposed to accomplish.

## `P0` Tokenizer

Primary artifacts:

- tokenizer model
- coverage diagnostics
- token-fragmentation report for Markdown, LaTeX, and tables

Exit criteria:

- acceptable delimiter preservation
- acceptable average tokens for formulas and complex tables

## `P1` Text Warmup

Primary artifacts:

- writer checkpoint
- held-out target perplexity report

Exit criteria:

- stable canonical generation behavior
- no obvious degradation on Markdown or LaTeX delimiters

## `P2` Multimodal Core

Primary artifacts:

- base parser checkpoint
- validation cards on exact internal slices
- auxiliary-head diagnostics

Exit criteria:

- stable page-to-Markdown training
- planner heads produce meaningful learning signal
- no collapse on formulas or tables

## `P3` Hard Curriculum

Primary artifacts:

- robustness checkpoint
- failure-slice report
- latency-aware quality comparison against `P2`

Exit criteria:

- improved hard-slice behavior
- no unacceptable regression on clean exact data
