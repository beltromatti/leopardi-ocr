# Frontier Synthesis

Date locked: 2026-04-07

This document consolidates the broader April 2026 frontier across VLMs, LLMs, training systems, inference systems, and compression into a single map for Leopardi.

It is deliberately curated rather than encyclopedic. The goal is not to list every new model release, but to identify the research directions that most plausibly change the Leopardi design space.

## Scope

This broader frontier pass focuses on the following questions:

- how the strongest open models are fitting more capability into fewer active parameters
- how compact VLMs are retaining useful multimodal quality
- how post-training and RL are shifting from vague alignment to verifiable objectives
- how runtime, kernel, and quantization choices are now feeding back into model design
- which directions matter specifically for a document parser that must be both exact and fast

## The Eight Frontier Trendlines

### 1. Active compute is replacing total parameters as the right optimization target

The strongest evidence comes from:

- `DeepSeek-V2`
- `DeepSeek-V3`
- `Kimi Linear`
- `Jamba`
- `Zamba`

What changed:

- the field is increasingly optimizing active parameters, KV footprint, and decode cost rather than only total model size
- MoE and KV-efficient attention are no longer niche tricks; they are now part of the main efficiency playbook
- hybrid attention and state-space designs remain relevant for long contexts and throughput-constrained deployments

Leopardi implication:

- the blueprint should track `active_params`, `KV_footprint`, `TTFT`, and `TPOT` from the start

### 2. Modern VLMs are moving toward stronger vision systems without sacrificing language quality

The strongest evidence comes from:

- `Qwen2.5-VL`
- `InternVL3`
- `Libra`
- `Qwen2.5-Omni`
- `LinguDistill`

What changed:

- stronger VLMs are no longer satisfied with a shallow projector on top of a frozen language model
- decoupled or routed visual pathways are becoming more attractive
- language-quality preservation is now an explicit research problem, not an accidental by-product

Leopardi implication:

- multimodal adaptation must not be allowed to damage the language backbone, because Markdown and LaTeX fidelity depend on it

### 3. Native-resolution and dynamic-resolution vision are now the default direction for dense visual parsing

The strongest evidence comes from:

- `PaddleOCR-VL`
- `Qwen2.5-VL`
- `InternVL3`
- `MindVL`

What changed:

- fixed-resolution pipelines are increasingly a liability on visually dense tasks
- test-time scaling and resolution adaptation are becoming practical levers
- document parsing, UI understanding, and chart-heavy inputs benefit from these changes more than generic captioning does

Leopardi implication:

- page complexity should drive visual budget decisions, not only raw image size

### 4. Compact VLMs are strategically real, not just demo artifacts

The strongest evidence comes from:

- `SmolVLM`
- `OpenELM`
- `Empirical Recipes for Efficient and Compact Vision-Language Models`
- `Firebolt-VL`

What changed:

- compact models are now competitive enough to matter for edge, local, and throughput-constrained serving
- architecture choices and data quality matter more than brute-force scale in these deployment regimes

Leopardi implication:

- a 0.9B to 3B multimodal family is a credible frontier target if the rest of the stack is strong

### 5. Data quality is becoming a deeper moat than raw pretraining scale

The strongest evidence comes from:

- `JEST`
- `Molmo and PixMo`
- `Visual Program Distillation`
- OCR-specific synthetic data engines such as `olmOCR`, `Infinity-Parser`, and `FireRed-OCR`

What changed:

- curated example selection, verified synthetic supervision, and high-quality open data are repeatedly delivering outsized gains
- high-quality datasets are increasingly treated as research contributions in their own right

Leopardi implication:

- a document-data engine is not support work; it is core model capability infrastructure

### 6. Post-training is shifting toward verifiable rewards and structurally checkable objectives

The strongest evidence comes from:

- `DeepSeek-R1`
- `HybridFlow`
- `verl`
- `TRL`
- OCR-side examples such as `olmOCR 2`, `FireRed-OCR`, and `Infinity-Parser`

What changed:

- post-training is moving away from preference-only pipelines when objective rewards are available
- RL infrastructure is becoming more modular and production-shaped
- exact-output tasks benefit disproportionately from verifiable rewards

Leopardi implication:

- Markdown validity, block structure, table shape, reading order, and LaTeX syntax should be treated as rewardable units

### 7. Inference is now a design constraint, not an afterthought

The strongest evidence comes from:

- `vLLM`
- `SGLang`
- `TensorRT-LLM`
- `FlashInfer`
- `PagedAttention`
- `Mooncake`

What changed:

- continuous batching, paged KV management, disaggregated serving, speculative decoding, and runtime-native quantization are all now baseline concerns
- model choices that do not map cleanly onto the serving stack incur a real competitive penalty

Leopardi implication:

- runtime compatibility must be part of architecture selection, not a later deployment task

### 8. Visual token reduction is becoming central for efficient VLM serving

The strongest evidence comes from:

- `LVPruning`
- `DivPrune`
- `ResPrune`

What changed:

- multiple recent papers show that a large fraction of visual tokens can be removed with small quality loss
- the field is moving from blunt token dropping to text-conditioned and structure-preserving pruning

Leopardi implication:

- adaptive visual budgets are mandatory for fast document parsing because not all pages deserve the same amount of compute

### 9. Structured decoding has become real systems infrastructure

The strongest evidence comes from:

- `XGrammar`
- `XGrammar 2`
- `llguidance`
- `JSONSchemaBench`

What changed:

- constrained decoding has moved from niche library code to runtime-integrated infrastructure
- exact-output systems no longer need to choose between validity and practical latency by default

Leopardi implication:

- exact Markdown and LaTeX generation should be treated as both a modeling and decoding problem

## Cross-Cutting Takeaways

### The current frontier is systems-shaped

The best results increasingly emerge from coordinated choices across:

- model architecture
- pretraining data
- post-training objective
- runtime
- kernels
- quantization
- evaluation

### The current frontier is modular, not monolithic

The most promising systems now often combine:

- a strong compact backbone
- routed or decoupled vision components
- selective post-training
- runtime-aware deployment targets

### Exactness has become a product of both modeling and verification

This is especially relevant for Leopardi because OCR-VL is an exact-output task. The wider frontier increasingly supports:

- verified synthetic supervision
- rewardable structural objectives
- targeted repair or specialist execution paths

## What Matters Most For Leopardi

### Likely high-value directions

- compact multimodal backbone with strong language preservation
- native or dynamic visual resolution
- adaptive visual token routing and pruning
- RLVR for structured output exactness
- grammar-aware structured decoding or repair
- first-class `vLLM` and `SGLang` compatibility
- quantization-aware deployment targets from the first training cycle

### Likely mistakes

- optimizing only total parameter count
- assuming one fixed image resolution or token budget
- relying only on SFT for exact Markdown and LaTeX
- postponing runtime and compression decisions until after modeling is done
- benchmarking only accuracy without latency and footprint

## Companion Documents

- `foundation-model-frontier-2026-04.md`
- `efficient-training-frontier-2026-04.md`
- `inference-systems-frontier-2026-04.md`
- `compression-and-efficiency-frontier-2026-04.md`
- `leopardi-blueprint-inputs-2026-04.md`
- `sources-frontier-2026-04.md`
