# Leopardi Pretraining Plan

Date locked: 2026-04-09

This document defines the pretraining plan for `Leopardi-S0 ~200M` (with
pretrained SigLIP2 vision encoder and SmolLM2-initialized decoder) and the
scale-up path to `Leopardi-S1 ~600M`.

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
The decoder is already initialized from SmolLM2-135M weights, so P1 adapts
those general language priors to the Leopardi tokenizer and canonical Markdown
output domain.

Implementation note:

- SmolLM2 initializes the decoder transformer stack itself
- Leopardi-native cross-attention remains random because it has no direct source counterpart
- token embeddings remain Leopardi-native when the tokenizer vocabulary differs from SmolLM2

Data:

- canonical Markdown targets from arXiv and PMC
- extracted LaTeX spans
- tables converted to canonical table blocks

What is trainable in P1:

- writer decoder only (~52M)
- SigLIP2 vision encoder and all novel components are frozen

Why:

- SmolLM2 provides general language priors but not Markdown/LaTeX fluency
- P1 bridges the gap: the decoder learns the new tokenizer vocabulary and
  the exact canonical output format before multimodal pressure
- this is faster than P1 in the previous design because SmolLM2 already
  has strong language patterns — P1 now mostly teaches the domain format

### P2. Core Multimodal Parsing Pretraining

Train the full model on paired page-to-canonical-target data.

What is trainable in P2:

- writer decoder (~77M, all layers)
- novel components: latent bottleneck + block planner + layout encoder (~27M)
- SigLIP2 vision encoder top 4 layers at low LR (~31M unfrozen out of 92.93M total)
- total trainable: ~135M out of ~199M

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
- preserve table, caption, and formula structure instead of learning a prose-only target

Auxiliary objectives:

- block-type prediction
- reading-order prediction
- orientation prediction
- table topology prediction
- formula span prediction

Learning rate strategy:

- SigLIP2 unfrozen layers: 1/10th of base learning rate
- novel components (bottleneck, planner, layout): base learning rate
- writer decoder: base learning rate
- this differential LR prevents catastrophic forgetting of pretrained vision features

Modern compact OCR-VLM results and repos also imply practical rules for `S0`:

- curriculum and sample weighting matter as much as raw data volume
- formula and table spans need extra token pressure, not only page-level supervision
- the writer and planner should usually move faster than the visual trunk on a single `RTX 5090`
- cheap layout side maps should be turned into training-visible memory, not treated as preprocessing only
- MTP-style future-token supervision is worth carrying early because it sharpens compact decoders and keeps serving options open

### P3. Hard-Case Curriculum

Once the model can parse clean born-digital pages, expose it aggressively to hard cases.

What is trainable in P3:

- ALL parameters unfrozen (~154M), including all SigLIP2 layers
- the vision encoder now fine-tunes fully to adapt to degraded/rotated/handwritten inputs

Data:

- synthetic corruptions derived from approved source sets
- handwriting corpora
- forms and receipts
- chart and figure subsets
- rotation-equivalent exact pairs built from exact core pages

Goal:

- robustness without sacrificing the clean-core parse ability

## Loss Design

### Main seq2seq loss

Use autoregressive next-token loss over canonical Markdown output.

### MTP loss

Use a small future-token objective on the same canonical targets.

Why:

- compact decoders benefit from stronger local lookahead pressure
- later speculative-serving or draft-style inference paths need the family to already support this behavior

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

Recommended S0 published pool (target ~10.3M total samples):

- `~5.31M` real-source samples
- `~500K` build-time multilingual synthetic pages from the Leopardi European generator
- `~4.5M` derived hard cases from `synthetic_from_exact`

Locked `S0` finetune follow-up:

- `F0-F3` together should stay at `1.50M` stage draws
- this is intentional hardening, not a second large-scale pretraining pass

Recommended pretraining exposure over the full S0 curriculum:

- `35%` exact paired pages from arXiv and PMC
- `30%` synthetic hard cases derived from exact sources
- `10%` formula-focused from UniMER-1M, CROHME, MathWriting, Im2LaTeX-100K
- `8%` table-focused from PubTables-1M, SciTSR, FinTabNet family
- `7%` layout-focused supervision from PubLayNet and DocLayNet
- `5%` multilingual from the Leopardi European generator DE/FR/ES/IT/PT
- `3%` handwriting from IAM, Bentham, READ 2016
- `2%` forms, receipts, charts from FUNSD, CORD, SROIE, ChartQA, PlotQA

Bundle usage by stage:

- `P1`
  - `tokenizer_v1`
  - `p1_text_warmup_v1`
- `P2`
  - `p2_exact_core_v1`
  - `p2_structural_aux_v1`
- `P3`
  - `p2_exact_core_v1`
  - `p2_structural_aux_v1`
  - `p3_hardcases_v1`

Inside those buckets, the compact-model recipe should explicitly oversample:

- `formula + rotation`
- `table + caption`
- `handwriting + structure`
- `european_multilingual + layout`

Text corruption pre-training (per MiniCPM-V 4.5):

- during P2: 10% of exact pages have text regions corrupted
- during P3: 20-40% corruption rate on hard-case pages
- target always stays exact — forces the model to use context

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

- single-GPU training on RTX 5090 (Blackwell, SM 12.0, 32GB VRAM)
- `bf16` training (Blackwell has strong bf16 tensor core support)
- gradient checkpointing on (required for 154M + activations in 32GB)
- `torch.compile` as the primary optimization path
- `F.scaled_dot_product_attention` for all attention (FlashAttention-compatible)
- CUDA 12.8+ with PyTorch 2.9+ (sm_120 support)
- Liger-Kernel fused ops where stable (RMSNorm, SwiGLU, cross-entropy, RoPE)

Recommended first training shape:

- page render DPI: mixed `144` and `192`
- SigLIP2-NaFlex handles dynamic resolution natively
- global batch via gradient accumulation (effective batch 32-64)
- cosine decay with short warmup
- module-wise learning-rate scaling:
  - SigLIP2 unfrozen layers: 1e-5
  - novel components: 1e-4
  - writer decoder: 1e-4

VRAM budget estimate for S0 (154M params):

- model parameters (bf16): ~0.4 GB
- optimizer states (AdamW, fp32): ~1.6 GB
- activations with gradient checkpointing: ~10-16 GB
- SigLIP2 frozen layers: ~0.17 GB (no grad)
- batch of page images + targets: ~2-4 GB
- estimated total: ~14-22 GB out of 32 GB available

Why this matters:

- the point of the `200M` phase is rapid algorithmic iteration, not one heroic training run
- the pretrained backbone means P2 converges much faster than from-scratch training

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

### 6. Do not let exact-core targets collapse into plain text

If exact-core conversion drops tables, captions, or display math, the model learns the wrong task.

## Scale-Up Path To `Leopardi-S1`

Once `Leopardi-S0` is stable and the best recipe is known:

- keep SigLIP2-base vision encoder (same for comparability)
- increase internal hidden from 576 to 960
- increase latent count from 192 to 384 (6 layers)
- increase decoder from 12 to 27 layers (initialized from SmolLM2-360M)
- increase planner from 3 to 5 layers and 64 to 112 block queries
- widen specialist adapters
- keep the same target representation and objectives
- increase MTP horizon from 2 to 3

The scale-up should be recipe-preserving, not architecture-resetting.
Every training stage (P0–P3) and finetuning stage (F0–F3) runs identically,
only with larger capacity and proportionally more data.
