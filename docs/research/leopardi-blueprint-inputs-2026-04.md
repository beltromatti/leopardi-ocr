# Leopardi Blueprint Inputs

Date locked: 2026-04-07

This document consolidates the broader frontier research into direct inputs for the final Leopardi blueprint.

## Non-Negotiable Design Inputs

### 1. Leopardi should be a document parser, not a plain OCR engine

This remains unchanged and is reinforced by the wider frontier.

### 2. Leopardi should optimize active compute, not only total parameter count

Relevant frontier signals:

- DeepSeek-V2 / V3
- Kimi Linear
- compact VLM recipes
- GLM-OCR
- PaddleOCR-VL-1.5

Implication:

- benchmark `active_params`, `visual_tokens`, `TTFT`, `TPOT`, and `KV footprint`

### 3. Leopardi should preserve language quality while specializing visually

Relevant frontier signals:

- InternVL3
- Libra
- LinguDistill
- Qwen2.5-Omni

Implication:

- include pure-text and language-intensive distillation phases

### 4. Leopardi should use adaptive visual compute

Relevant frontier signals:

- native-resolution VLMs
- LVPruning
- DivPrune
- ResPrune

Implication:

- avoid a fixed visual token budget

### 5. Leopardi should be designed around runtime reality

Relevant frontier signals:

- vLLM
- SGLang
- FlashInfer
- Mooncake
- DeepEP
- XGrammar
- llguidance

Implication:

- runtime compatibility is part of model design, not a separate stage

## Recommended Blueprint Shape

### Base family

- compact multimodal backbone in the 0.9B to 3B class
- native-resolution or dynamic-resolution vision pathway
- strong language core with explicit preservation strategy
- option for a decoupled or routed visual expert path

### Optional expertization

- math expert
- table expert
- handwriting expert
- chart/diagram expert

### Post-training

- SFT for structured Markdown + LaTeX
- RLVR / GRPO-style structural optimization
- selective distillation for language and reasoning preservation

### Runtime

- first-class support for `vLLM` and `SGLang`
- adaptive token routing
- speculative or MTP-friendly decode path
- structured decoding path for exact-output modes
- quantization-aware serving plan from the beginning

## Key Research Gaps Leopardi Can Exploit

### 1. Exact Markdown remains under-optimized

### 2. Joint handling of handwriting + layout + formulas is still weak

### 3. Graphics-aware parsing is still underdeveloped outside dots.mocr

### 4. Public benchmark plus public training plus public serving is still rare in OCR-VL

### 5. Many strong systems still report weak latency cards

### 6. The broader frontier still underexploits grammar-aware exact-output RL for document parsing

### 7. Exact Markdown plus LaTeX remains underexplored as a constrained-decoding problem

## Blueprint Constraints Implied by the Frontier

- do not assume one architecture family will dominate every regime
- do not assume giant general VLMs are the best path
- do not postpone systems work
- do not postpone compression work
- do not build evaluation around only one benchmark family

## Final Blueprint Questions To Resolve Next

1. Dense compact model or sparse compact model?
2. Pure VLM parser or routed hybrid parser?
3. Single decoder or shared decoder plus experts?
4. Which runtime is primary: vLLM, SGLang, or dual-target?
5. Which quantization target is mandatory for v1: FP8, W8A8, W4A16, or multiple?
6. Which benchmark bundle is required for every release candidate?
