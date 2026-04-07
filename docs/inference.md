# Leopardi Inference Plan

Date locked: 2026-04-07

This document defines the serving and decoding blueprint for Leopardi.

The inference goal is not only low latency.
It is low latency without giving away exactness on the hard slices that matter.

## Inference Philosophy

Leopardi should not process every page with the same budget.

The correct strategy is:

- spend little on easy pages
- escalate only when needed
- repair locally whenever possible

## Runtime Targets

### Primary baseline

- `vLLM`

Why:

- safest general open serving baseline
- strong quantization and multimodal support
- structured outputs are now a first-class server feature, not an afterthought

### High-performance target

- `SGLang`

Why:

- strong structured-output posture
- strong rollout and high-throughput serving path
- proven relevance for RL and post-training loops

### Kernel layer

- `FlashInfer`

Use when:

- custom performance work becomes necessary
- quantized or MoE-like future variants need deeper optimization

## Page Serving Pipeline

### Step 1. Render and canonicalize

- render page
- estimate global orientation
- normalize geometry and contrast
- produce lightweight side maps

### Step 2. Complexity estimate

Estimate page complexity from cheap signals:

- visual density
- number of suspected blocks
- formula density
- table density
- handwriting likelihood
- chart likelihood

### Step 3. Route into one of three modes

#### `fast`

Use when:

- clean page
- low density
- no clear specialist trigger

Budget:

- small visual token budget
- one pass
- validation only, repair only on hard failure

#### `standard`

Use when:

- moderate density
- some specialist hints
- uncertain planner output

Budget:

- medium visual token budget
- one local repair pass allowed

#### `hard`

Use when:

- formulas
- merged-cell tables
- handwriting
- charts
- photographed distortions

Budget:

- larger crop budget
- specialist adapters enabled
- stronger constrained decoding
- localized multi-pass repair allowed

## Decode Strategy

### First-pass decode

Use:

- block planner
- writer decoder
- light grammar constraints where cheap
- runtime-native structured output features when they do not distort latency comparisons

Do not:

- force the heaviest grammar constraints globally on every page

### Repair decode

Use:

- stricter grammar constraints
- block-local context
- specialist hints

This is where `xgrammar`, `llguidance`, and runtime-native grammar backends become most attractive.

## Structured Decoding Policy

The right blueprint is hybrid, not pure decode-time enforcement everywhere.

### Global decode

Use constrained decoding only where overhead is low and reliability is high:

- block start and end markers
- fenced block closure
- simple table forms
- formula delimiter closure
- regex or grammar-constrained line families supported by the serving backend

### Local repair

Use stronger grammar enforcement on:

- complex table blocks
- malformed lists
- malformed code fences
- invalid LaTeX spans

Why this is the right compromise:

- full-document grammar constraints can become expensive
- local repair gives most of the exactness benefit at lower latency

### Runtime-specific rule

For `vLLM`:

- use the structured-outputs interface with backend selection left on `auto` unless a benchmark explicitly pins a backend
- treat grammar-vs-regex-vs-choice changes as decode-policy changes that must be logged

For `SGLang`:

- use native structured outputs for regex, JSON-like validation, and grammar-constrained repair where the backend shows a real latency advantage
- benchmark it as a separate runtime condition, not as an invisible optimization

## Quantization Roadmap

### `Leopardi-S0`

Serving targets:

- `bf16` reference
- `fp8` if stable
- `w4a16` or similar compact deployment mode

### `Leopardi-S1`

Serving targets:

- `fp8`
- stronger KV-cache quantization
- possibly low-bit activation-aware modes if they preserve exactness

Tooling:

- `TorchAO`
- `llm-compressor`

## Latency Targets

The exact numbers will be discovered experimentally, but the direction is fixed.

### `Leopardi-S0`

Must target:

- meaningful page-level latency leadership in the `<150M` class
- practical throughput on one `RTX 5090`

### `Leopardi-S1`

Must target:

- absolute frontier latency-quality tradeoff against the best `~1B` class competitors

## Document Assembly

The parser output is page-based internally, but the final product is document-level.

The document assembler should handle:

- duplicate header and footer suppression
- page break policy
- cross-page section continuity
- carry-over of continued tables when detectable
- final Markdown packaging

This stays outside the core model for the `100M` phase.

That is a feature, not a compromise.

## What The Inference Stack Must Log

Every production and benchmark run must log:

- page complexity tier
- chosen mode
- visual token budget
- output tokens
- repair count
- latency breakdown
- validator failures

Without this, we cannot optimize intelligence per parameter honestly.
