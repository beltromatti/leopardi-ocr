# Leopardi Directions

Date locked: 2026-04-07

This document synthesizes the competitor papers, benchmarks, and actual codebases into concrete guidance for Leopardi.

## What The Field Is Converging Toward

### 1. Small OCR specialists are beating giant general VLMs

The strongest document parsers are no longer just the biggest models.

- PaddleOCR-VL-1.5: 0.9B
- GLM-OCR: 0.9B
- HunyuanOCR: 1B
- MinerU2.5 / MonkeyOCR-pro-1.2B: 1.2B
- OpenDoc / UniRec: 0.1B class

Implication for Leopardi:

- compactness is now a frontier advantage, not a compromise

### 2. Pure end-to-end is not the whole story

The winning systems are split between:

- end-to-end page parsers
- hybrid layout + region recognition systems
- parser + verifier or parser + repair stacks

Implication for Leopardi:

- do not lock the architecture to pure OCR-free or pure modular
- keep routing and specialist experts as first-class options

### 3. RL is moving from general reasoning into OCR structure control

Current frontier examples:

- FireRed-OCR: format-constrained GRPO
- olmOCR 2: unit-test rewards on synthetic benchmark cases
- Infinity-Parser: layout-aware RL
- MonkeyOCR v1.5: visual consistency RL for tables
- GLM-OCR: stable full-task RL claim

Implication for Leopardi:

- reward functions should target structure validity, table integrity, reading order, and LaTeX syntax

### 4. Benchmarks are becoming product-shaped

The most useful benchmark families now resemble real deployment requirements:

- OmniDocBench for page parsing
- Real5 / photographed document settings
- olmOCR-Bench for difficult PDF-to-Markdown linearization
- MDPBench for multilingual photographed parsing
- specialist formula/table benchmarks

Implication for Leopardi:

- benchmark selection is no longer an academic detail
- it defines what "best in the world" even means

### 5. Deployment systems are a competitive moat

The best repos now expose:

- vLLM
- SGLang
- Ollama or MLX
- ONNX or TensorRT
- CPU/GPU split for layout and recognition
- speculative decoding or multi-token prediction

Implication for Leopardi:

- speed leadership will come from systems design as much as model quality

## What Competitors Still Do Poorly

### 1. Latency reporting is inconsistent

Many frontier papers still say "fast" without:

- fixed hardware
- batch size
- page complexity mix
- retry policy
- token budget

Leopardi opportunity:

- publish a rigorous latency card for every model and every benchmark

### 2. Markdown validity is still under-specified

Most systems optimize text similarity and benchmark scores, but not a strict Markdown contract.

Leopardi opportunity:

- define a Markdown AST and grammar constraints
- verify output validity block by block
- repair locally instead of re-decoding the whole page

### 3. Math is still not truly solved in integrated parsers

Integrated VLM parsers improved, but formula exactness still lags specialist systems.

Leopardi opportunity:

- combine page parser and math specialist
- verify LaTeX syntactically and semantically
- train with math-specific rewards and hard negatives

### 4. Handwriting plus layout plus math remains weak as a joint problem

Most repos are strong at two of these, not all three together.

Leopardi opportunity:

- target the intersection:
  - rotation
  - handwriting
  - tables
  - formulas
  - photographed scans

### 5. Graphics are underexploited

dots.mocr is the main warning sign. Most others still underinvest in chart and diagram understanding.

Leopardi opportunity:

- treat charts/graphics as first-class parse targets
- emit SVG or internal graph representations when useful
- translate them into Markdown and structured sidecar assets

## Concrete Leopardi Architecture Guidance

### Keep three interchangeable parse modes

1. `fast_path`
   - compact single-pass parser for easy pages
2. `expert_path`
   - layout-aware parsing with math/table/handwriting specialists
3. `repair_path`
   - localized verification and correction

### Make the target representation stricter than competitors

Required internal representation:

- block types
- reading order
- normalized Markdown
- normalized LaTeX spans
- provenance boxes and confidence

### Build rewardable units

Leopardi should be trainable against automatic rewards for:

- markdown validity
- block closure
- equation syntax
- table structure
- reading-order consistency
- latency-aware quality

### Treat data generation as a product

Competitors repeatedly win through data factories:

- FireRed geometry + semantics
- Infinity-Doc synthetic + real
- olmOCR HTML template mining and synth bench
- MonkeyDoc and MDPBench data pipelines

Leopardi should have:

- PDF-source pairs
- synthetic distortions
- handwriting overlays
- hard table and formula synthesis
- benchmark failure mining

## Best Code References By Problem

### End-to-end benchmark discipline

- `olmOCR`

### Industrial deployment and breadth

- `PaddleOCR`

### Compact self-hosted parser stack

- `GLM-OCR`

### Tiny modular specialist design

- `OpenOCR / OpenDoc / UniRec`

### PDF-to-Markdown engineering

- `OCRFlux`
- `MonkeyOCR`
- `MinerU`

### Math subsystem

- `UniMERNet`

### Layout-aware RL direction

- `Infinity-Parser`

## Immediate Build Recommendations

### Phase A

- implement benchmark harness for OmniDocBench, olmOCR-Bench, and MDPBench-style input
- define Markdown AST and validators
- build latency measurement harness with fixed hardware cards

### Phase B

- train a compact 0.9B to 1.5B page parser baseline
- add a formula specialist
- add table structure verification

### Phase C

- introduce reward-based structural finetuning
- add hard-page routing
- add chart/diagram parsing extensions

