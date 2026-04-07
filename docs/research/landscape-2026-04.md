# Research Landscape Snapshot

Date locked: 2026-04-07

This snapshot is intentionally focused on ideas that matter for a top-tier single-page OCR system with Markdown + LaTeX output.

## Signals That Matter Right Now

### OCR-free parsers are now the baseline to beat

- `OLMOCR` shows that VLM-based PDF parsing can scale to massive PDF corpora and produce structured text directly from page images.
- `MinerU2.5` pushes a decoupled high-resolution document parser, which is relevant for accuracy on dense pages without exploding inference cost.
- `GOT-OCR 2.0` reinforces the case for unified OCR across scenes, documents, and structured outputs rather than brittle detector-recognizer stacks.

### Specialist heads still matter on hard regions

- `UniMERNet` and related formula recognizers remain relevant because math exactness is much less forgiving than plain-text OCR.
- Table structure recovery is still best treated as a first-class problem, not a formatting afterthought.
- Handwriting remains a failure mode for many general document VLMs, so dedicated data and routing are necessary.

### Benchmarks have moved beyond plain text accuracy

- `OmniDocBench` evaluates general-purpose document parsing with multimodal structure.
- `olmOCR Bench` stresses PDF-to-markdown extraction quality.
- `Real5-OmniDocBench` raises the bar with harder, more realistic document conditions.

## Leopardi Design Consequences

- The primary model should emit Markdown-native blocks, not plain text with post-hoc formatting.
- Math must be decoded into LaTeX with specialist supervision and syntax-aware verification.
- Speed requires conditional computation: draft fast, zoom only where needed, repair locally.
- Evaluation must score structure, formulas, and latency together.

## Source Links

- Nougat: https://arxiv.org/abs/2308.13418
- GOT-OCR 2.0: https://arxiv.org/abs/2409.01704
- MinerU2.5: https://wangbindl.github.io/publications/MinerU2_5.pdf
- OLMOCR: https://openreview.net/forum?id=kQ2GMikZpW
- OmniDocBench: https://github.com/opendatalab/OmniDocBench
- olmOCR: https://github.com/allenai/olmocr
- UniMERNet: https://arxiv.org/abs/2404.15215

