# Competitor Landscape

Date locked: 2026-04-07

## Scope

This landscape only includes competitors relevant to Leopardi's target:

- single-page PDF or page image input
- structured output, ideally Markdown or equivalent
- formulas rendered as LaTeX or high-fidelity structured text
- support for tables, mixed layouts, and difficult documents
- practical deployment constraints, not pure benchmark accuracy alone

## Market Segments

### 1. Specialized End-to-End Document VLMs

These are the most direct competitors.

- PaddleOCR-VL / PaddleOCR-VL-1.5
- HunyuanOCR
- FireRed-OCR
- GLM-OCR
- MonkeyOCR / MonkeyOCR v1.5
- dots.ocr / dots.mocr
- OCRVerse
- DeepSeek-OCR
- MinerU2.5
- olmOCR / olmOCR 2
- OpenDoc-0.1B
- OCRFlux
- Infinity-Parser
- Chandra OCR

### 2. Classical or Hybrid Pipelines

Still competitive, especially in narrow or latency-sensitive settings.

- PP-StructureV3
- Marker
- Docling
- PaddleOCR pipeline family
- RapidTable and other component specialists

### 3. Commercial OCR APIs

These matter because users compare against them in production.

- Mistral OCR
- Mathpix
- Google/Gemini document parsing via general VLM or document stack
- OpenAI general VLMs

### 4. General VLMs Used as OCR

Not optimized for OCR, but very strong on some benchmarks.

- Gemini 2.5 Pro
- Gemini 2.0 Flash
- GPT-4.1
- GPT-4o
- Qwen2.5-VL / Qwen3-VL
- InternVL3.x
- Claude 3.7 Sonnet and Claude Sonnet 4 on OCR-only leaderboards

## Architectural Patterns

### Pattern A: OCR-free page-to-Markdown models

Examples:

- PaddleOCR-VL
- HunyuanOCR
- DeepSeek-OCR
- dots.mocr
- olmOCR

Advantages:

- simpler output path
- fewer hand-engineered modules
- often better at preserving global context

Weaknesses:

- can hallucinate structure
- can overuse language priors
- may struggle when fine-grained region specialization is required

### Pattern B: Modular or semi-modular parsing

Examples:

- MonkeyOCR
- GLM-OCR
- OpenDoc-0.1B
- PP-StructureV3

Advantages:

- better controllability
- often better speed for easy pages
- easier to debug per module

Weaknesses:

- error propagation
- more deployment complexity

### Pattern C: Constrained or RL-enhanced structural generation

Examples:

- FireRed-OCR with format-constrained GRPO
- olmOCR 2 with RLVR and unit-test rewards
- MonkeyOCR v1.5 with visual consistency RL for tables
- OCRVerse with two-stage SFT-RL

Advantages:

- directly addresses malformed structure
- better fit for Markdown and LaTeX validity

Weaknesses:

- more training complexity
- reward design can overfit benchmark formatting

## The Most Important Competitors

### PaddleOCR-VL-1.5

- strongest public all-around benchmark signal
- compact 0.9B footprint
- explicitly extends into real-world physical distortions via Real5-OmniDocBench

### HunyuanOCR

- strongest current multilingual and wild-document vendor evidence
- 1B end-to-end OCR specialist VLM
- a major deployment-minded competitor if its published numbers reproduce externally

### dots.mocr

- the most differentiated competitor for Leopardi's chart and graphics goal
- parses both text and graphics into unified text/code representations
- especially dangerous if future versions improve page-level structural fidelity without sacrificing graphic parsing

### FireRed-OCR

- best current example of turning a general VLM into an OCR specialist through data factory plus constrained RL
- important because Leopardi will likely need similar structure-validity training

### GLM-OCR

- compact 0.9B model with explicit throughput optimization through multi-token prediction
- important benchmark for the speed front

### MonkeyOCR v1.5 / pro models

- strongest public speed/accuracy transparency among the modern open competitors
- SRR decomposition is a useful counter-example to pure end-to-end design

### olmOCR 2

- best openly documented English PDF linearization specialist with an evaluation philosophy very close to Leopardi's product output
- its unit-test reward framework is directly reusable as a design pattern

## Where Competitors Win Today

### Parsing accuracy

- PaddleOCR-VL class
- HunyuanOCR
- FireRed-OCR

### English PDF-to-Markdown stress tests

- dots.mocr
- Chandra OCR
- Infinity-Parser
- olmOCR 2 family

### Multilingual coverage

- PaddleOCR-VL claims 109 languages
- HunyuanOCR claims 100+ languages
- dots.ocr/dots.mocr aims at broad multilingual coverage
- MDPBench now exposes how weak many open models still are outside dominant scripts

### Formula handling

- Mathpix remains a production reference
- UniMERNet remains the strongest open specialist signal
- PaddleOCR-VL, HunyuanOCR, FireRed-OCR, and MonkeyOCR are the main integrated competitors

### Graphics and charts

- dots.mocr is currently the clearest leader among open systems
- OCRVerse is also relevant due to explicit vision-centric OCR

### Efficiency

- OpenDoc-0.1B / UniRec-0.1B for small-footprint specialist design
- PaddleOCR-VL and GLM-OCR for compact frontier quality
- MonkeyOCR-pro-1.2B for disclosed runtime efficiency

## Leopardi Design Implications

### We need structured decoding, not plain OCR

The frontier has moved from text extraction to structured page serialization. Markdown-native decoding is mandatory.

### We need math as a first-class subsystem

Integrated VLM OCR is still not enough. Formula exactness remains brittle and needs specialist supervision, decoding constraints, and verification.

### We need robustness beyond born-digital PDFs

Real5-style distortions, photographed pages, handwriting, and rotation are now benchmarked explicitly. Training only on clean PDFs will fail.

### We need a chart/graphics story

Most competitors still treat graphics as second-class. dots.mocr shows this is now a frontier opportunity, not a niche.

### We need latency-aware routing

Compact 0.9B to 1.2B specialists are now competitive with or better than much larger general VLMs. Conditional computation is strategically correct.

