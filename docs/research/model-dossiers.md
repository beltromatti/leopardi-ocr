# Model Dossiers

Date locked: 2026-04-07

## PaddleOCR-VL / PaddleOCR-VL-1.5

- Class: specialized document VLM
- Size: 0.9B
- Architecture: NaViT-style dynamic-resolution visual encoder plus ERNIE language model
- Training signal: multilingual document parsing with complex elements including tables, formulas, and charts
- Public strengths: strongest public page parsing evidence, strong multilingual support, compact footprint
- Main threat to Leopardi: compact high-accuracy baseline that does not require giant general VLM scale
- Main open question: standardized public pages-per-second data is still weak in the first-pass source set

## HunyuanOCR

- Class: end-to-end OCR expert VLM
- Size: 1B
- Architecture: native ViT plus lightweight LLM via MLP adapter
- Training signal: OCR-specific data plus RL strategies; supports spotting, parsing, IE, translation
- Public strengths: very strong multilingual and in-the-wild numbers in official report
- Main threat to Leopardi: end-to-end breadth without huge parameter count
- Main risk: strongest numbers are still mostly vendor-authored

## FireRed-OCR

- Class: specialized structural OCR framework built on a general VLM
- Size: not clearly disclosed in the abstract beyond its Qwen3-VL basis
- Architecture: specialized Qwen3-VL-based parser
- Training signal: Geometry + Semantics Data Factory, multi-task pre-alignment, specialized SFT, format-constrained GRPO
- Public strengths: 92.94 OmniDocBench v1.5, explicit anti-hallucination focus
- Main threat to Leopardi: directly optimizes structural validity, which matches our Markdown objective

## GLM-OCR

- Class: compact multimodal OCR model
- Size: 0.9B
- Architecture: 0.4B CogViT visual encoder plus 0.5B GLM decoder
- Training signal: structured OCR tasks with multi-token prediction and two-stage layout plus recognition pipeline
- Public strengths: explicit throughput optimization via MTP, compact footprint
- Main threat to Leopardi: one of the most credible speed-focused public architectures

## MonkeyOCR / MonkeyOCR v1.5

- Class: modular document parsing VLM
- Size: 3B base, 1.2B compact variant, larger v1.5 framework
- Architecture: Structure-Recognition-Relation triplet paradigm
- Training signal: MonkeyDoc 4.5M bilingual instances; v1.5 adds visual-consistency RL for tables and cross-page table modules
- Public strengths: unusually transparent speed tables and strong accuracy
- Main threat to Leopardi: strong engineering tradeoff between modular control and modern VLM recognition

## dots.ocr / dots.mocr

- Class: multimodal OCR parsing text plus graphics
- Size: 3B
- Architecture: compact VLM trained to emit textual and code-level representations for text and graphics
- Training signal: PDFs, rendered webpages, native SVG assets, staged pretraining and SFT
- Public strengths: strongest open signal on graphics-aware parsing, 83.9 on olmOCR-Bench, high OCR Arena Elo
- Main threat to Leopardi: broadens the target beyond OCR into image-to-code document understanding

## OCRVerse

- Class: holistic OCR VLM
- Size: 4B in benchmark tables
- Architecture: end-to-end OCR spanning text-centric and vision-centric OCR
- Training signal: cross-domain data engineering and two-stage SFT-RL
- Public strengths: unified treatment of documents, charts, web pages, and plots
- Main threat to Leopardi: strong cross-domain generalization if page-level structure keeps improving

## DeepSeek-OCR

- Class: compressed-context OCR VLM
- Size: 3B with DeepSeek3B-MoE-A570M decoder
- Architecture: DeepEncoder plus MoE decoder, optimized for high visual compression
- Training signal: optical 2D mapping and compressed token regime
- Public strengths: strong efficiency story and competitive benchmark results
- Main threat to Leopardi: compression-driven low-token parsing can unlock better latency
- Main risk: independent analysis shows high dependence on language priors under corruption

## MinerU2.5

- Class: decoupled high-resolution document parser
- Size: 1.2B
- Architecture: decoupled VLM for efficient high-resolution parsing
- Training signal: document parsing with emphasis on high-resolution efficiency
- Public strengths: very strong OmniDocBench results with moderate model size
- Main threat to Leopardi: high-resolution accuracy without giant model cost

## olmOCR / olmOCR 2

- Class: PDF linearization VLM plus toolkit
- Size: 7B
- Architecture: specialized OCR VLM for linearized structured text extraction
- Training signal: 260k pages from 100k crawled PDFs for olmOCR; RLVR with unit-test rewards for olmOCR 2
- Public strengths: benchmark maintainer for olmOCR-Bench, strong English PDF-to-text stress performance, explicit cost reporting
- Main threat to Leopardi: mature evaluation philosophy and reward design very close to our product needs

## OpenDoc-0.1B / UniRec-0.1B

- Class: ultra-lightweight document parser
- Size: 0.1B
- Architecture: PP-DocLayoutV2 plus UniRec-0.1B unified recognition
- Training signal: UniRec40M for text and formulas, rebuilt for text plus formulas plus tables in OpenDoc
- Public strengths: remarkable accuracy for size, direct evidence that tiny specialists remain viable
- Main threat to Leopardi: small-footprint deployment benchmark

## OCRFlux

- Class: PDF-to-Markdown VLM toolkit
- Size: 3B
- Architecture: lightweight multimodal toolkit for Markdown conversion and cross-page merging
- Training signal: vendor-authored benchmark suite plus task-specific merge tasks
- Public strengths: high page-granular self-benchmark EDS and cross-page merging support
- Main threat to Leopardi: practical pipeline engineering around Markdown conversion
- Main risk: benchmark evidence is mostly self-authored

## Infinity-Parser

- Class: scanned document parser with layout-aware RL
- Size: 7B in public repo naming
- Architecture: VLM parser trained with layout-aware reinforcement learning
- Training signal: Infinity-Doc-55K and later Infinity-Doc-400K synthetic plus real data
- Public strengths: strong olmOCR-Bench results and explicit structural-fidelity focus
- Main threat to Leopardi: RL on layout fidelity for scanned documents

## Chandra OCR

- Class: structured OCR model and hosted API
- Size: not clearly disclosed in first-pass public sources
- Architecture: markdown/html/json output preserving layout
- Training signal: not sufficiently disclosed in first-pass sources
- Public strengths: 83.1 on olmOCR-Bench table and strong handwriting/table claims
- Main threat to Leopardi: very strong practical PDF-to-Markdown competitor
- Main risk: architecture and training transparency is currently thin

## Mistral OCR

- Class: commercial OCR API
- Size: undisclosed
- Architecture: undisclosed; current docs expose `mistral-ocr-latest`
- Training signal: undisclosed
- Public strengths: production API, Markdown output, table/html options
- Main threat to Leopardi: low-friction commercial adoption
- Main risk: weaker public benchmark evidence than the best open specialized systems

## Mathpix

- Class: commercial STEM OCR API
- Size: undisclosed
- Architecture: undisclosed
- Training signal: undisclosed
- Public strengths: strong production reputation for math and STEM Markdown/LaTeX conversion
- Main threat to Leopardi: formulas remain one of the hardest domains and Mathpix is still a reference point

## General VLMs

### Gemini 2.5 Pro / Gemini 2.0 Flash

- Strength: excellent OCR-only scores on IDP leaderboard and strong competitive performance on OmniDocBench-style tasks
- Weakness: undisclosed architecture/training for OCR specialization, usually less controllable than dedicated parsers

### GPT-4.1 / GPT-4o

- Strength: strong OCR-only and formula performance, broad multimodal utility
- Weakness: cost and weaker document-specialist benchmark performance than the best dedicated systems

### Qwen2.5-VL / Qwen3-VL / InternVL3.x

- Strength: useful base models and strong general VLM baselines
- Weakness: they are increasingly outperformed by compact OCR specialists on structured parsing tasks
