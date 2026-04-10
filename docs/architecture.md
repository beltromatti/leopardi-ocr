# Leopardi Architecture

Date locked: 2026-04-09

This document defines the revised research blueprint for Leopardi, updated to
integrate pretrained vision and language components following the April 2026
architecture revision (see `docs/research/architecture-revision-2026-04.md`).

The current implementation surface for this blueprint lives in:

- `model/`
- `src/leopardi/model/`
- `configs/model/leopardi_s0.yaml`

Leopardi is a full-document parser whose external contract is:

- input: PDF documents
- internal processing unit: page
- output: Markdown-first structured parse
- formulas: LaTeX
- tables: Markdown-native canonical table format

The first research vehicle is `Leopardi-S0`, a `~200M` parameter model built on
pretrained vision and language backbones, optimized for fast iteration on `RTX 5090`.
The final product model will be `Leopardi-S1`, a `~600M` scale-up of the same
design after the right recipe is found.

## Core Design Principles

### 1. Spend parameters on semantics, not on cheap geometry problems

Leopardi should not waste scarce model capacity on tasks that can be handled cheaply and deterministically:

- page rotation
- coarse skew
- contrast normalization
- page border cleanup
- duplicate header and footer detection at document assembly time

For a `~200M` model, this is non-negotiable.

### 2. Optimize active compute, not only total parameter count

The `200M` phase is explicitly about intelligence per parameter and intelligence per millisecond.

That means:

- dense compact backbone first
- dynamic visual token budgeting
- blockwise decoding instead of unconstrained page-long autoregression
- explicit reuse of cheap layout side maps inside the model memory path
- localized repair instead of full re-decode

### 3. The model should write a structured document, not transcribe a flat string

Markdown and LaTeX exactness are central product requirements.

Leopardi therefore uses:

- explicit block planning
- block-type-conditioned writing
- grammar-aware decoding and repair
- syntax validators during training and inference

### 4. The 200M model must be a research instrument, not just a small product model

`Leopardi-S0` is designed to maximize ablation speed on a single `RTX 5090`.
It must be small enough to retrain frequently and instrumented enough to reveal what actually works.

## Model Family

### `Leopardi-S0`

Research vehicle for rapid iteration.

Target size:

- `~200M` total parameters
- dense model
- no MoE in v1
- pretrained vision encoder: SigLIP2-base-patch16-NaFlex (`92.93M` vision parameters by HF config)
- pretrained decoder initialization: SmolLM2-135M weight transfer

Why pretrained backbones:

- every competitive system in the 100M–700M class uses pretrained components
- SigLIP2 provides strong document-aware visual features from day one
- SmolLM2 provides language generation priors from 2T tokens of pretraining
- the Leopardi innovations (planner, bottleneck, side-maps, repair) differentiate
  on top of proven foundations

Why dense first:

- better single-GPU training behavior
- simpler runtime
- easier quantization
- fewer routing confounders during research

### `Leopardi-S1`

Final-scale product model after the recipe is locked.

Target size:

- `~600M` total parameters
- same architecture family
- SigLIP2-base vision encoder (same as S0 for comparability)
- larger decoder initialized from SmolLM2-360M
- more depth, more latent capacity, optional lightweight routed specialists

## External System Shape

Leopardi is not a single forward pass glued directly to a PDF.

The end-to-end system has five layers:

1. `document_ingest`
   - PDF render and page extraction
2. `page_canonicalizer`
   - cheap geometry cleanup and orientation normalization
3. `page_parser`
   - the compact VLM core
4. `validator_repair`
   - syntax and structural verification with localized repair
5. `document_assembler`
   - cross-page deduplication, ordering, and final Markdown packaging

The model is the core, but the system wins only if all five layers are disciplined.

## Output Contract

### Leopardi Markdown Canonical Form

Leopardi should emit a canonical representation, not arbitrary Markdown style variation.

The canonical form is:

- headings as ATX headings
- paragraphs as plain blocks
- lists as standard Markdown lists
- code blocks as fenced blocks
- inline math as `$...$`
- display math as `$$...$$`
- figures as Markdown image-style or figure blocks with captions
- simple rectangular tables as GitHub-flavored Markdown tables
- complex merged-cell tables as a Markdown-native fenced `table` block

Example for a complex table:

````md
```table
caption: Revenue by segment
columns: 4
cells:
  - [0, 0, 1, 0, "Region"]
  - [0, 1, 0, 2, "Q1"]
  - [0, 3, 0, 4, "Q2"]
  - [1, 1, 1, 1, "2024"]
  - [1, 2, 1, 2, "2025"]
```
````

Reason:

- standard Markdown cannot represent rowspan and colspan faithfully
- this keeps the external output Markdown-native while remaining exact

## `Leopardi-S0` Architecture

### Parameter Budget

Target allocation:

- pretrained vision encoder (SigLIP2-base-NaFlex): `92.93M`
- pixel shuffle and projection: `~1.77M`
- structural latent bottleneck: `~13.39M`
- block planner: `~13.33M`
- layout side-map encoder: `~0.3M`
- writer decoder (12 layers, 576 hidden, SmolLM2-initialized): `~77.36M`
- MTP heads and auxiliary heads: `~0.5M`

Target total:

- `~199M` loaded total, reported as the `~200M` S0 class

### 1. Page Canonicalizer

This stage is intentionally cheap and mostly non-neural.

Responsibilities:

- render page at one or more DPIs
- estimate global orientation
- estimate coarse skew
- normalize contrast
- remove heavy margins and page borders when safe
- produce lightweight side maps:
  - line density map
  - probable text-direction map

Why it matters:

- saves model capacity
- reduces token waste
- improves robustness to arbitrary rotation and photographed pages

### 2. Pretrained Visual Encoder (SigLIP2-base-NaFlex)

The visual front-end is a pretrained SigLIP2 ViT-B/16 with NaFlex
dynamic-resolution support.

Specifications:

- model: `google/siglip2-base-patch16-naflex`
- architecture: ViT-B/16 with 2D learned positional embeddings
- hidden size: 768
- layers: 12
- attention heads: 12
- patch size: 16
- parameters: `92.93M` for the vision tower
- NaFlex: processes images at native aspect ratio with variable sequence length

Post-encoder processing:

- pixel shuffle (2×2→1) to reduce visual token count by 4×
- MLP projection from 768 to internal hidden dimension (`576` for S0, `960` for S1)

Why SigLIP2-base-NaFlex:

- state-of-the-art pretrained vision encoder as of 2026
- NaFlex excels specifically on OCR and document retrieval tasks
- 92.93M parameters provide strong visual features that would require
  billions of training samples to learn from scratch
- same architecture family scales to SigLIP2-Large (303M) for S1 if needed
- widely validated in production VLMs (Granite Vision, SmolVLM2)

Training regime:

- freeze bottom 8 layers during P2 core multimodal pretraining
- fine-tune top 4 layers from the start of P2
- progressively unfreeze during P3 and finetuning stages

### 2b. Layout Side-Map Encoder

The canonicalizer already computes cheap structural hints.
`Leopardi-S0` should convert those hints into trainable memory, not discard them.

Blueprint:

- ingest line-density and text-direction maps
- encode them with a lightweight convolutional side branch
- pool them into a small fixed grid of layout tokens (3×4 = 12 tokens)
- feed those tokens into the latent bottleneck and writer memory

Parameters: ~0.3M

Why it matters:

- adds layout awareness without paying for a second large vision tower
- improves robustness on rotated, multi-column, handwritten, and irregular pages
- increases intelligence per parameter by turning deterministic geometry into reusable context

### 3. Structural Latent Bottleneck

Instead of exposing the decoder to all visual tokens directly, Leopardi compresses the page into a small set of learned parse latents.

Blueprint for S0:

- `192` learned latents at hidden dimension 576
- `3` cross-attention layers (self-attention + cross-attention to visual+layout tokens + FFN)
- latent outputs feed both the planner and the writer

Parameters: ~13.39M

Blueprint for S1:

- `384` learned latents at hidden dimension 960
- `6` cross-attention layers

Research backing:

- InternVL-X (2025): SOTA with only 20% visual tokens, validating aggressive compression
- Perceiver/Perceiver IO: learned latent cross-attention is effective for multimodal compression
- Q-Former (BLIP-2): similar cross-attention bottleneck widely validated

Why it matters:

- improves intelligence per parameter
- reduces decoder burden
- gives a natural place for block planning and specialist probing
- keeps long pages from exploding decode cost

### 4. Block Planner

Leopardi should not decode the entire page as one undifferentiated stream.

Blueprint for S0: 3 layers, 64 query slots, hidden 576 (~13.33M)
Blueprint for S1: 5 layers, 112 query slots, hidden 960 (~61.59M)

The planner predicts an ordered sequence of block descriptors:

- heading
- paragraph
- list
- table
- figure
- figure_caption
- equation
- page_header
- page_footer
- marginalia

Each planned block also predicts:

- approximate source region
- expected content length bucket
- confidence
- expert hint:
  - default
  - math
  - table
  - handwriting
  - chart

Why it matters:

- reduces decode entropy
- improves reading order
- localizes repair
- makes document assembly tractable

### 5. Writer Decoder

The writer is a modern autoregressive decoder conditioned on:

- structural latents
- layout-side tokens
- current block descriptor (from planner)
- block-local crop features when needed
- previously emitted blocks

The writer emits canonical Markdown plus LaTeX.

Blueprint for S0: 12 layers, hidden 576, GQA 9Q/3KV, SwiGLU, RoPE, RMSNorm (~77.36M)
Blueprint for S1: 27 layers, hidden 960, GQA 15Q/5KV, SwiGLU, RoPE, RMSNorm (~373.95M)

Modern architectural features (adopted from Qwen3, 2025):

- **RoPE** (Rotary Position Embeddings) with theta=1000000 for unlimited length extrapolation
- **GQA** (Grouped Query Attention) for efficient KV-cache at inference
- **SwiGLU** activation in FFN for better approximation capacity
- **RMSNorm** with pre-normalization for faster and stabler training
- **QK-Norm** for training stability at scale
- **No QKV bias** (following Qwen3 empirical finding)

Initialization:

- Self-attention and FFN weights initialized from SmolLM2-135M (S0) or SmolLM2-360M (S1)
  via layer selection and dimension projection
- Cross-attention layers randomly initialized
- Token embeddings randomly initialized (domain-specific tokenizer)
- This gives the decoder strong pretrained language patterns while allowing
  full customization of the cross-modal conditioning and output vocabulary

Memory construction:

```
memory = cat(structural_latents, layout_tokens, planner_states)
```

Each WriterBlock applies:

1. Causal self-attention (with RoPE)
2. Cross-attention to memory
3. SwiGLU FFN

The writer also exposes native multi-token-prediction heads (horizon=2 for S0, 3 for S1).
This keeps the family compatible with speculative-serving and draft-style acceleration.

### 6. Specialist Adapters and Heads

At `~200M`, Leopardi should not fork into many independent models.

The right compromise is:

- one shared backbone
- lightweight specialist adapters or heads

Required specialist paths:

- `math_path`
  - formula span boundary prediction
  - LaTeX-heavy decoding bias
- `table_path`
  - grid topology head
  - cell adjacency and span prediction
- `handwriting_path`
  - stronger local-text decoding bias for cursive/noisy lines
- `chart_path`
  - chart text block extraction and axis/legend grouping

For `Leopardi-S0`, these should be lightweight adapters, not large side models.

### 7. Verifier and Local Repair

Validation is part of the architecture, not a post-hoc convenience.

Required validators:

- Markdown syntax validator
- LaTeX syntax validator
- table cell-span validator
- reading-order validator
- duplicate block detector

Repair policy:

- never re-decode the full page unless necessary
- repair only invalid or low-confidence blocks
- allow stronger grammar constraints during repair than during the first pass

## Why This Architecture Is Better Suited To 200M Than The Main Alternatives

### Not a plain OCR pipeline

Why reject:

- too many moving parts
- hard to make Markdown-first
- difficult to optimize jointly for speed and exactness

### Not a giant general VLM

Why reject:

- impossible to iterate rapidly on a single `RTX 5090`
- weak signal-to-noise for research
- too much capacity wasted on non-document tasks

### Not a small MoE first

Why reject for v1:

- routing overhead is expensive at small scale
- single-GPU training and serving become less clean
- quantization and debugging become harder

### Not trained from scratch

Why reject at this scale:

- every successful system at 100M–700M uses pretrained components
- a 9.4M vision encoder from scratch cannot compete with 92.93M SigLIP2
- a 50M decoder from scratch cannot match SmolLM2 trained on 2T tokens
- pretrained components reduce data requirements by orders of magnitude

### Why pretrained-backbone block-planned dense VLM is the right first bet

- SigLIP2 provides strong visual features from day one
- SmolLM2 initialization provides strong language generation priors
- block planning reduces decode entropy and enables localized repair
- the novel components (bottleneck, planner, side-maps) differentiate on
  top of proven foundations rather than competing with them
- efficient enough for rapid iteration on a single RTX 5090
- compatible with constrained decoding
- naturally scales to `600M`

## Training Hooks Designed Into The Architecture

The model must support the following training signals from day one:

- block-type classification
- reading-order supervision
- box-to-block alignment
- table structure supervision
- formula span supervision
- line-orientation supervision
- multi-token prediction on canonical targets
- text-only continuation on canonical Markdown outputs

This is deliberate.
The `200M` model has to learn from many cheap auxiliary signals because that is how small models become disproportionately strong.

## Inference Hooks Designed Into The Architecture

The model must support three inference modes without architecture changes:

### `fast`

- single pass
- small visual token budget
- minimal repair

### `standard`

- adaptive crops
- one repair pass on invalid blocks

### `hard`

- larger crop budget
- specialist-heavy routing
- stronger constrained decoding

## Scaling Path To `Leopardi-S1`

The `600M` model should not be a redesign.

Scale the same ingredients with the same vision encoder:

| Dimension | S0 (200M) | S1 (600M) |
|-----------|-----------|-----------|
| Vision encoder | SigLIP2-base 92.93M | SigLIP2-base 92.93M |
| Internal hidden | 576 | 960 |
| Latent bottleneck layers | 3 | 6 |
| Latent count | 192 | 384 |
| Planner layers | 3 | 5 |
| Planner queries | 64 | 112 |
| Decoder layers | 12 | 27 |
| Decoder init source | SmolLM2-135M | SmolLM2-360M |
| MTP horizon | 2 | 3 |

What should stay unchanged:

- SigLIP2-base vision encoder (for perfect comparability)
- page canonicalizer
- canonical Markdown contract
- block planner and writer split
- localized repair philosophy
- benchmark protocol

If S1 needs stronger vision, SigLIP2-Large (303M) can be swapped in as a controlled ablation.

## Non-Negotiable Architectural Decisions

1. Pretrained vision encoder (SigLIP2 family)
2. Dense compact first, not MoE first
3. Block planning before writing
4. Markdown canonical form with exact table extension
5. Dynamic visual budget
6. Validation and local repair as first-class modules
7. Same design family for `200M` and `600M`
8. RoPE, GQA, SwiGLU, RMSNorm in all transformer blocks
9. `torch.compile`-compatible forward pass for RTX 5090
