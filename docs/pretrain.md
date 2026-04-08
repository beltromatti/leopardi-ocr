# Leopardi Pretraining Plan

Date locked: 2026-04-07

This document defines the pretraining plan for `Leopardi-S0` and the scale-up path to `Leopardi-S1`.

The current implementation surface for this plan now lives in:

- `pretraining/`
- `src/leopardi/pretraining/`
- `configs/pretraining/`
- `configs/runtime/train_rtx5090.yaml`

The pretraining objective is not “learn generic vision-language intelligence”.
It is:

- maximize parsing intelligence per parameter
- maximize exact structured output capability
- preserve training speed on `RTX 5090`

## Pretraining Strategy Summary

Leopardi pretraining should happen in four stages.

### P0. Tokenizer and Target Vocabulary

Train the tokenizer on canonical targets, not on generic web text.

Source text for tokenizer training:

- canonical Markdown from arXiv-derived pairs
- canonical Markdown from PMC-derived pairs
- LaTeX formulas from arXiv, PMC, CROHME, MathWriting, and Im2LaTeX-100K
- table canonical blocks from PubTables-1M and SciTSR conversions

Tokenizer requirements:

- preserve Markdown delimiters well
- preserve LaTeX delimiters and common commands well
- avoid fragmenting table syntax excessively

Recommended first tokenizer:

- SentencePiece unigram
- `32k` to `48k` vocabulary

### P1. Text-Only Domain Warmup

Before multimodal training, warm up the writer decoder on target-domain text.

Data:

- canonical Markdown targets from arXiv and PMC
- extracted LaTeX spans
- tables converted to canonical table blocks

Why:

- a `100M` model cannot afford weak output-language priors
- this improves exact Markdown and LaTeX generation without using a large general LM

### P2. Core Multimodal Parsing Pretraining

Train the full model on paired page-to-canonical-target data.

Primary data:

- arXiv
- PMC

Auxiliary supervision:

- PubLayNet
- DocLayNet
- PubTables-1M
- SciTSR

Main objective:

- page to canonical Markdown plus LaTeX transduction

Auxiliary objectives:

- block-type prediction
- reading-order prediction
- orientation prediction
- table topology prediction
- formula span prediction

Modern compact OCR-VLM results and repos also imply three practical rules for `S0`:

- curriculum and sample weighting matter as much as raw data volume
- formula and table spans need extra token pressure, not only page-level supervision
- the writer and planner should usually move faster than the visual trunk on a single `RTX 5090`

### P3. Hard-Case Curriculum

Once the model can parse clean born-digital pages, expose it aggressively to hard cases.

Data:

- synthetic corruptions derived from approved source sets
- handwriting corpora
- forms and receipts
- chart and figure subsets

Goal:

- robustness without sacrificing the clean-core parse ability

## Loss Design

### Main seq2seq loss

Use autoregressive next-token loss over canonical Markdown output.

### Planner loss

Supervise:

- block types
- reading order
- block length bucket
- expert hint

### Table losses

Supervise:

- cell adjacency
- row and column grouping
- merged-cell spans

### Formula losses

Supervise:

- formula boundary detection
- exact LaTeX sequence decoding

### Robustness losses

Supervise:

- rotation class
- line-direction hints
- handwriting/noise difficulty prediction

## Data Mixture For `Leopardi-S0`

Recommended initial mixture:

- `45%` exact paired pages from arXiv and PMC
- `20%` synthetic corruptions of exact paired pages
- `10%` layout-focused supervision from PubLayNet and DocLayNet
- `10%` table-focused supervision from PubTables-1M and SciTSR
- `10%` formula-focused supervision from CROHME, MathWriting, and Im2LaTeX-100K
- `5%` handwriting, forms, and chart-heavy tasks

This should be treated as the first strong prior, not as immutable truth.
Weak or partially teacher-derived supervision should be discounted explicitly rather than mixed at full strength.

## Curriculum

### Stage A

- clean born-digital pages only
- low-resolution variation only

### Stage B

- synthetic distortions
- arbitrary rotations
- multi-column and dense-table oversampling

### Stage C

- handwriting overlays
- photographed and scanned documents
- receipts, forms, chart-heavy pages

The model should not start on the hardest data.
For small models that usually wastes training budget.

## Training Mechanics For `RTX 5090`

Primary assumptions:

- single-GPU training
- `bf16`
- gradient checkpointing on
- FlashAttention-compatible attention path
- Liger kernels where stable

Recommended first training shape:

- page render DPI: mixed `144` and `192`
- adaptive crop budget instead of always higher resolution
- global batch via gradient accumulation
- cosine decay with short warmup and module-wise learning-rate scaling

Why this matters:

- the point of the `100M` phase is rapid algorithmic iteration, not one heroic training run

## What Not To Do In Pretraining

### 1. Do not start with huge multilingual synthetic corpora

That is not the first bottleneck.

### 2. Do not train the first model only on OCR crops

Leopardi is a parser, not a recognizer-only system.

### 3. Do not overfit to benchmark distributions

Use benchmark-style data, but do not collapse pretraining into benchmark imitation.

### 4. Do not introduce MoE in the first training phase

It obscures the architecture search.

### 5. Do not use a flat token loss for all structures

Formulas and complex tables dominate the frontier error surface and need explicit token-level emphasis.

## Scale-Up Path To `Leopardi-S1`

Once `Leopardi-S0` is stable and the best recipe is known:

- increase encoder depth and width
- increase latent count
- increase decoder capacity
- widen specialist adapters
- keep the same target representation and objectives

The scale-up should be recipe-preserving, not architecture-resetting.
