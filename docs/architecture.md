# Leopardi Architecture

Date locked: 2026-04-08

This document defines the final research blueprint for Leopardi.

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

The first research vehicle is `Leopardi-S0`, a compact `~100M` parameter model optimized for fast iteration on `RTX 5090`.
The final product model will be `Leopardi-S1`, a `~500M` scale-up of the same design after the right recipe is found.

## Core Design Principles

### 1. Spend parameters on semantics, not on cheap geometry problems

Leopardi should not waste scarce model capacity on tasks that can be handled cheaply and deterministically:

- page rotation
- coarse skew
- contrast normalization
- page border cleanup
- duplicate header and footer detection at document assembly time

For a `~100M` model, this is non-negotiable.

### 2. Optimize active compute, not only total parameter count

The `100M` phase is explicitly about intelligence per parameter and intelligence per millisecond.

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

### 4. The 100M model must be a research instrument, not just a small product model

`Leopardi-S0` is designed to maximize ablation speed on a single `RTX 5090`.
It must be small enough to retrain frequently and instrumented enough to reveal what actually works.

## Model Family

### `Leopardi-S0`

Research vehicle for rapid iteration.

Target size:

- `90M` to `110M` total parameters
- dense model
- no MoE in v1

Why dense first:

- better single-GPU training behavior
- simpler runtime
- easier quantization
- fewer routing confounders during research

### `Leopardi-S1`

Final-scale product model after the recipe is locked.

Target size:

- `450M` to `550M` total parameters
- same architecture family
- more depth, more latent capacity, larger visual budget, optional lightweight routed specialists

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

- visual stem and hierarchical encoder: `34M` to `40M`
- latent compressor and planner: `10M` to `14M`
- writer decoder: `42M` to `48M`
- auxiliary heads and embeddings: `8M` to `12M`

Target total:

- `~100M`, with the first concrete preset currently in the low-`90M` range

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

### 2. Adaptive Visual Tokenizer

The visual front-end should be a compact hierarchical encoder with dynamic-resolution support.

Blueprint:

- convolutional patch stem for local robustness
- hierarchical ViT-style encoder with patch merging
- variable patch density based on page complexity
- optional crop refinement for dense local regions
- side-map reuse for layout priors without a second heavy encoder

Why this design:

- better local inductive bias than a plain ViT at this size
- lower token count than naive high-resolution patching
- better fit for document pages than generic image classification backbones

### 2b. Layout Side-Map Encoder

The canonicalizer already computes cheap structural hints.
`Leopardi-S0` should convert those hints into trainable memory, not discard them.

Blueprint:

- ingest line-density and text-direction maps
- encode them with a lightweight convolutional side branch
- pool them into a small fixed grid of layout tokens
- feed those tokens into the latent bottleneck and writer memory

Why it matters:

- adds layout awareness without paying for a second large vision tower
- improves robustness on rotated, multi-column, handwritten, and irregular pages
- increases intelligence per parameter by turning deterministic geometry into reusable context

### 3. Structural Latent Bottleneck

This is one of the most important parts of the blueprint.

Instead of exposing the decoder to all visual tokens directly, Leopardi compresses the page into a small set of learned parse latents.

Blueprint:

- `128` to `256` learned latents
- several rounds of cross-attention from latents to visual tokens
- latent outputs feed both the planner and the writer

Why it matters:

- improves intelligence per parameter
- reduces decoder burden
- gives a natural place for block planning and specialist probing
- keeps long pages from exploding decode cost

### 4. Block Planner

Leopardi should not decode the entire page as one undifferentiated stream.

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

The writer is a compact autoregressive decoder conditioned on:

- structural latents
- layout-side tokens
- current block descriptor
- block-local crop features when needed
- previously emitted blocks

The writer emits canonical Markdown plus LaTeX.

Why an autoregressive writer remains the right choice:

- output is heterogeneous and hierarchical
- formulas and tables benefit from exact sequence control
- structured decoding and repair can be applied at block level

To keep this efficient at serving time, the writer should also expose native multi-token-prediction heads.
This keeps the family compatible with speculative-serving and draft-style acceleration paths used by modern runtimes.

### 6. Specialist Adapters and Heads

At `~100M`, Leopardi should not fork into many independent models.

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

## Why This Architecture Is Better Suited To 100M Than The Main Alternatives

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

### Why block-planned dense VLM is the right first bet

- best match to exact-output transduction
- efficient enough for rapid iteration
- compatible with constrained decoding
- naturally scales to `500M`

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
The `100M` model has to learn from many cheap auxiliary signals because that is how small models become disproportionately strong.

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

The `500M` model should not be a redesign.

Scale the same ingredients:

- deeper and wider visual encoder
- richer layout-side memory
- more structural latents
- stronger writer decoder
- larger block planner
- optional routed specialists for math and tables

What should stay unchanged:

- page canonicalizer
- canonical Markdown contract
- block planner and writer split
- localized repair philosophy
- benchmark protocol

## Non-Negotiable Architectural Decisions

1. Dense compact first, not MoE first
2. Block planning before writing
3. Markdown canonical form with exact table extension
4. Dynamic visual budget
5. Validation and local repair as first-class modules
6. Same design family for `100M` and `500M`
