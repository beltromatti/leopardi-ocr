# Foundation-Model Frontier

Date locked: 2026-04-07

This document extends Leopardi research beyond OCR-specific competitors into the broader frontier of compact, efficient, and highly capable LLMs and VLMs as of April 2026.

Source registry for this pass: `sources-frontier-2026-04.md`

## Why This Matters for Leopardi

Leopardi will not win only by copying document parsers. The frontier is being shaped by:

- more intelligence per active parameter
- better data efficiency
- more aggressive inference optimization
- stronger multimodal grounding with smaller models
- system co-design across model, kernel, runtime, and serving stack

OCR-VL now sits inside a much larger ecosystem of ideas.

## Executive Takeaways

### 1. Dense models are no longer the default answer

The most important efficiency patterns now include:

- MoE with lower active parameter count
- latent-attention or compressed-attention variants
- hybrid attention and state-space architectures
- native-resolution or dynamic-resolution vision encoders
- adaptive token pruning and visual token bypass

### 2. Training quality is shifting from raw scale to curated scale

The strongest recent systems repeatedly combine:

- better data selection
- synthetic task generation
- curriculum design
- RL with verifiable rewards
- selective distillation instead of indiscriminate imitation

### 3. Inference is becoming a first-class architecture constraint

Models are now designed with serving in mind:

- multi-token prediction
- speculative decoding
- KV-cache reduction
- disaggregated prefill/decode serving
- kernel specialization and paged memory

### 4. Compact VLMs are becoming strategically credible

The field is clearly moving toward:

- compact VLMs that can still parse dense visual scenes
- deployment-oriented stacks for document, UI, and agent tasks
- preservation of linguistic capability during multimodal adaptation

This is directly relevant to Leopardi.

## Architectural Directions

### MoE + KV-Efficient Attention

Best current open signal:

- `DeepSeek-V2` and `DeepSeek-V3`

Why they matter:

- MoE reduces active parameter count at inference
- MLA compresses KV cache heavily
- multi-token prediction connects training objective to inference acceleration
- auxiliary-loss-free load balancing reduces MoE quality tax

Leopardi implication:

- sparse compute is credible if routing is stable and serving stack is ready
- KV efficiency matters for long documents and region-rich pages

### Hybrid Transformer + SSM / Linear Attention

Best current signals:

- `Jamba`
- `Zamba`
- `Kimi Linear`
- `Firebolt-VL`

Why they matter:

- they attack the quadratic attention bottleneck directly
- they improve long-context memory behavior
- they may be especially attractive when OCR-style tasks require long structural outputs or heavy prefix reuse

Leopardi implication:

- a hybrid decoder family should stay on the table
- linear or hybrid attention becomes more attractive if Markdown + LaTeX outputs get long

### Native-Resolution and Dynamic-Resolution Vision

Best current signals:

- `PaddleOCR-VL`
- `Qwen2.5-VL`
- `InternVL3`
- `MindVL`

Why they matter:

- fixed tiling often wastes compute or loses layout fidelity
- native-resolution processing is increasingly standard for visually dense tasks
- test-time resolution search is becoming a practical lever

Leopardi implication:

- document parsing should not be built around a naive fixed-resolution assumption
- adaptive resolution must be part of the design from day one

### Linguistic Preservation in VLMs

Best current signals:

- `InternVL3` native multimodal pretraining
- `Libra` decoupled vision system
- `LinguDistill`
- `Qwen2.5-Omni` Thinker-Talker split for modality separation

Why they matter:

- multimodal adaptation can damage core language capability
- smaller VLMs are especially sensitive to cross-modal interference

Leopardi implication:

- if Leopardi is adapted from a strong LM, we should actively preserve language quality
- selective distillation and modular multimodal interfaces are not optional details

### Adaptive Visual Token Reduction

Best current signals:

- `LVPruning`
- `DivPrune`
- `ResPrune`
- broader compact-VLM optimization recipes

Why they matter:

- inference cost in VLMs is often dominated by visual tokens
- fixed token budgets are increasingly seen as suboptimal

Leopardi implication:

- document pages vary wildly in density
- token budget should depend on page complexity, not only on image size

## Frontier Models and Papers That Matter Most

### LLM / Core Architecture

- `DeepSeek-V2`: MoE + MLA, strong cost-efficiency signal
- `DeepSeek-V3`: stronger MoE system co-design, FP8 training, MTP, distillation from R1
- `Jamba`: hybrid Transformer-Mamba MoE
- `Zamba`: compact SSM-transformer hybrid
- `Kimi Linear`: strong long-context efficiency and throughput signal
- `OpenELM`: open and compact language modeling with efficient layer-wise scaling

### VLM / MLLM

- `Qwen2.5-VL`: strong open general-purpose VLM and OCR-friendly base
- `InternVL3`: native multimodal pretraining and test-time recipes
- `Libra`: decoupled and routed vision-system direction
- `Qwen2.5-Omni`: streaming multimodal architecture with modality-aware decomposition
- `MindVL`: efficient multimodal training system and native-resolution design
- `Firebolt-VL`: linear-time decoder direction for compact VLMs
- `SmolVLM`: compact open VLM family with strong deployment relevance
- `Empirical Recipes for Efficient and Compact VLMs`: deployment-focused analysis rather than pure scaling
- `LinguDistill`: practical recovery of linguistic ability in VLMs

### Data and Distillation

- `JEST`: joint example selection for multimodal pretraining
- `Molmo and PixMo`: open-data quality and training-pipeline signal
- `Visual Program Distillation`: distilling tools and programmatic reasoning into VLMs
- `ViPER`: self-bootstrapped visual perception evolution

## What This Means for Leopardi

### The likely winning base is not a giant general VLM

The broader frontier still points toward:

- a compact but high-quality multimodal backbone
- dynamic visual compute
- structured decoding
- specialist experts where exactness matters

### OCR-VL should borrow from both document parsing and general MLLM research

Leopardi should combine:

- OCR-specialist supervision
- compact-VLM systems engineering
- language-model preservation techniques
- RLVR-style structural rewards

### Document parsing is becoming a systems problem

The model alone will not be enough. The strongest ideas now span:

- architecture
- training recipe
- quantization
- kernel stack
- serving runtime
- evaluation protocol
