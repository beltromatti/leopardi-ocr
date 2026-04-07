# Open-Source Codebase Audit

Date locked: 2026-04-07

This audit is based on the actual repositories vendored as submodules under `external/competitors/`, not just on papers or benchmark tables.

## Why This Matters

Two competitors may look similar on a leaderboard but be radically different engineering references:

- one may ship a full training stack, benchmark harness, and deployable parser
- another may only publish model weights plus a thin inference script

Leopardi needs both views:

- who is strongest in the market
- who is actually teaching us something implementable

## Taxonomy

### Complete Frameworks

Repos with substantial training, evaluation, deployment, and documentation surfaces:

- PaddleOCR
- OpenOCR
- olmOCR

### SDK + Finetuning Stacks

Repos with strong deployability plus at least partial finetuning instructions:

- GLM-OCR
- UniMERNet

### Production Toolkits

Repos focused on robust parsing and deployment, with weaker public training openness:

- MinerU
- MonkeyOCR
- OCRFlux
- Infinity-Parser

### Inference Releases

Repos mostly exposing model usage or demos:

- HunyuanOCR
- DeepSeek-OCR
- dots.ocr
- FireRed-OCR
- Chandra

### Minimal Release

- OCRVerse

## Repo-by-Repo Findings

### PaddleOCR

- Path: `external/competitors/paddleocr`
- Maturity: complete framework
- Signals:
  - massive breadth across `ppocr/`, `ppstructure/`, `deploy/`, `benchmark/`, `tests/`, `tools/`
  - open training stack, deployment stack, benchmarking scripts, ONNX/TensorRT/serving paths
  - explicit benchmark and latency documentation
- What Leopardi learns:
  - industrial pipeline discipline still matters
  - evaluation and deployment are treated as first-class, not afterthoughts
- Limitation:
  - codebase breadth is huge, which can slow iteration and hide the exact "best path" for a frontier parser

### HunyuanOCR

- Path: `external/competitors/hunyuanocr`
- Maturity: inference release
- Signals:
  - repo centers on `Hunyuan-OCR-master/Hunyuan-OCR-vllm/run_hy_ocr.py`
  - strong README coverage of usage and benchmark claims
  - little exposed training code
- What Leopardi learns:
  - clean model-release packaging and strong task prompting matter
- Limitation:
  - weak reproducibility for training and ablations

### GLM-OCR

- Path: `external/competitors/glm-ocr`
- Maturity: SDK + finetuning stack
- Signals:
  - real package `glmocr/`, CLI/server paths, frontend/backend apps
  - self-hosted pipeline with layout analysis plus parallel recognition
  - explicit vLLM and SGLang speculative decoding configs
  - `examples/finetune/` includes LoRA and full SFT via LLaMA-Factory
- What Leopardi learns:
  - compact model plus throughput-oriented inference is a serious design center
  - layout on CPU while GPU serves OCR is a strong systems idea
- Limitation:
  - RL details are described in paper/README more than implemented openly in repo

### FireRed-OCR

- Path: `external/competitors/firered-ocr`
- Maturity: inference release
- Signals:
  - repo is mostly README plus inference scripts
  - strongest value is conceptual: format-constrained GRPO and data factory
- What Leopardi learns:
  - structural validity needs to be optimized explicitly
- Limitation:
  - no real open training/eval stack to reproduce

### OCRVerse

- Path: `external/competitors/ocrverse`
- Maturity: minimal release
- Signals:
  - effectively README and assets only in the current public repo snapshot
- What Leopardi learns:
  - strategic idea is important, code reference is weak
- Limitation:
  - not usable as an engineering template

### DeepSeek-OCR

- Path: `external/competitors/deepseek-ocr`
- Maturity: inference release
- Signals:
  - nested release package with `DeepSeek-OCR-vllm`
  - explicit vLLM and Transformers inference paths
  - no open training stack in repo snapshot
- What Leopardi learns:
  - compressed-context OCR can be productized cleanly
- Limitation:
  - repo is closer to model release than research platform

### MonkeyOCR

- Path: `external/competitors/monkeyocr`
- Maturity: production toolkit
- Signals:
  - practical parser entrypoint in `parse.py`
  - modular code in `magic_pdf/`
  - good deployment, quantization, demos, and unusually transparent speed tables
- What Leopardi learns:
  - the SRR split remains one of the strongest modular alternatives to pure end-to-end parsing
  - speed disclosure should be treated as a product feature
- Limitation:
  - less open training code than the README and papers would suggest

### dots.ocr

- Path: `external/competitors/dots-ocr`
- Maturity: inference release
- Signals:
  - demos for HF, vLLM, Streamlit, batch parsing, and SVG variants
  - core parser and model inference modules are open
  - no major public training stack in repo snapshot
- What Leopardi learns:
  - structured graphics parsing should sit in the same product family as OCR
- Limitation:
  - training and ablation openness is limited

### olmOCR

- Path: `external/competitors/olmocr`
- Maturity: complete framework
- Signals:
  - serious package structure in `olmocr/`
  - open benchmark suite, train configs, synthetic data tooling, tests, Elo scripts
  - explicit GRPO/GRPO-like RL and synthetic benchmark generation in docs
- What Leopardi learns:
  - benchmark-driven OCR development is a major advantage
  - page-granular PDF to markdown supervision format is operationally clean even when the product target is full-document parsing
  - reward engineering based on unit tests and synthetic benchmarks is a strong frontier direction
- Limitation:
  - 7B base is less favorable for compact-latency deployment than 0.9B to 1.2B specialists

### OCRFlux

- Path: `external/competitors/ocrflux`
- Maturity: production toolkit
- Signals:
  - lean package with `ocrflux/inference.py`, client, server, and dense eval scripts
  - strong emphasis on PDF-to-Markdown and cross-page merging
- What Leopardi learns:
  - practical Markdown post-processing and merge logic is a competitive feature
- Limitation:
  - benchmark universe is still vendor-shaped

### OpenOCR

- Path: `external/competitors/openocr`
- Maturity: complete framework
- Signals:
  - full training/eval/infer stack under `openrec/`, `opendet/`, `tools/`, `configs/`, `docs/`
  - OpenDoc and UniRec are integrated into one ecosystem
- What Leopardi learns:
  - tiny specialist systems can be built in a disciplined, reproducible way
  - separate detector/recognizer/toolkit abstractions remain valuable
- Limitation:
  - less focused on a single frontier page parser than PaddleOCR-VL or olmOCR

### MinerU

- Path: `external/competitors/mineru`
- Maturity: production toolkit
- Signals:
  - strong CLI/server stack, rich docs, demos, and project ecosystem
  - practical parsing pipeline, but training code is not the primary public artifact
- What Leopardi learns:
  - product surface and documentation quality matter for adoption
- Limitation:
  - current repo is stronger as a parser toolkit than as a fully open research stack

### Chandra

- Path: `external/competitors/chandra`
- Maturity: inference release
- Signals:
  - polished package, tests, and benchmark assets
  - clear local/HF/vLLM modes
- What Leopardi learns:
  - a narrow, polished parser can compete hard on PDF-to-Markdown
- Limitation:
  - training openness is weak

### INF-MLLM / Infinity-Parser

- Path: `external/competitors/inf-mllm`
- Maturity: production toolkit plus broader multimodal repo
- Signals:
  - contains `Infinity-Parser`, `Infinity-Synth`, and the broader INF-MLLM codebase
  - strong inference and dataset exposure, RL direction is prominent in README
- What Leopardi learns:
  - layout-aware RL and synthetic data generation are a serious scanned-doc direction
- Limitation:
  - the repo is broader than the parser itself, so the OCR-specific workflow is less focused than olmOCR

### UniMERNet

- Path: `external/competitors/unimernet`
- Maturity: SDK + specialist training stack
- Signals:
  - formula-specific train/eval/demo stack with CDM metric tooling
  - open configs, scripts, and dataset framing
- What Leopardi learns:
  - formula OCR remains specialist enough to justify dedicated assets, metrics, and model variants
- Limitation:
  - this is a subsystem reference, not a full document parser

## Strategic Conclusions

### The best reusable code references are not the same as the best leaderboard entries

Best engineering templates:

- olmOCR
- GLM-OCR
- PaddleOCR
- OpenOCR

Best product pressure:

- PaddleOCR-VL-1.5
- HunyuanOCR
- dots.mocr
- FireRed-OCR

### Most repos are still weaker on open training than on open inference

That leaves a major opportunity for Leopardi:

- build a truly open benchmark + training + inference stack for Markdown + LaTeX page parsing

### The field is splitting into two camps

- compact, deployable OCR specialists around 0.1B to 1.2B
- broader multimodal OCR systems that absorb charts, graphics, screens, and code-like outputs

Leopardi should not choose only one. It should unify them through routing and verification.
